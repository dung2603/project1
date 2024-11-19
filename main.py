
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.model import CombinedDepthModel
from dataloader import DataLoadPreprocess
from evaluation import compute_metrics
from loss import DepthLoss
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device=None,
        num_epochs=100,
        min_depth=0.1,
        max_depth=10.0,
        early_stopping_patience=15,
        initial_lr=1e-4,
        max_grad_norm=1.0,
        validation_interval=0.25
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm
        self.validation_interval = validation_interval

        self.optimizer = optim.Adam(self.model.parameters(), lr=initial_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.criterion = DepthLoss(alpha=1.0, beta=0.1, delta=0.1).to(self.device)  # Sử dụng DepthLoss

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.writer = SummaryWriter('runs/depth_estimation_experiment')

    def _process_batch(self, batch, training=True):
        images = batch['image'].to(self.device)
        depths = batch['depth'].to(self.device)
        masks = batch.get('mask', None)
        if masks is not None:
            masks = masks.to(self.device)

        if depths.dim() == 3:
            depths = depths.unsqueeze(1)
        if masks is not None and masks.dim() == 3:
            masks = masks.unsqueeze(1)

        if training:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(training):
            depth_map = self.model(images)
            epsilon = 1e-6
            depth_map = F.softplus(depth_map) + epsilon  # Đảm bảo giá trị độ sâu dương

            depth_map = self._resize_depth_map(depth_map, depths)

            mask = self._mask_depth(depths)
            if masks is not None:
                mask = mask * masks  # Kết hợp mask từ dữ liệu và depth range

            # Áp dụng mask
            depth_map = depth_map * mask
            depths = depths * mask

            # Tính loss
            total_loss = self.criterion(depth_map, depths)

            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            return {'total_loss': total_loss.item(), **compute_metrics(depth_map, depths, mask=mask)}

    def _resize_depth_map(self, depth_map, target_depths):
        if depth_map.shape != target_depths.shape:
            depth_map = F.interpolate(depth_map, size=target_depths.shape[2:], mode='bilinear', align_corners=True)
        return depth_map

    def _mask_depth(self, target):
        return ((target > self.min_depth) & (target < self.max_depth)).float()

    def train_epoch(self, epoch):
        print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
        train_losses = []
        num_batches = len(self.train_loader)
        interval = max(1, int(num_batches * self.validation_interval))

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            self.global_step = epoch * num_batches + batch_idx
            losses = self._process_batch(batch, training=True)
            train_losses.append(losses['total_loss'])
            self.writer.add_scalar('Training/Loss/Total', losses['total_loss'], self.global_step)

            if (batch_idx + 1) % interval == 0:
                self.validate(is_final=(batch_idx + 1 == num_batches))

        avg_train_loss = np.mean(train_losses)
        print(f"\nAverage Training Loss: {avg_train_loss:.4f}")

        if num_batches % interval != 0:
            self.validate(is_final=True)

    def validate(self, is_final=False):
        if self.test_loader is None:
            print("No test set provided.")
            return

        self.model.eval()
        val_losses, val_metrics = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation"):
                metrics = self._process_batch(batch, training=False)
                val_losses.append(metrics.pop('total_loss'))
                val_metrics.append(metrics)

        avg_val_loss = np.mean(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if val_metrics:
            avg_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
            for k, v in avg_metrics.items():
                print(f"  {k}: {v:.4f}")
                self.writer.add_scalar(f'Validation/Averages/{k}', v, self.global_step)

        if is_final:
            self.scheduler.step()
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint('best_model.pth')
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.epochs_no_improve} epochs with no improvement.")
                    raise StopIteration

    def train(self):
        try:
            for epoch in range(self.num_epochs):
                self.train_epoch(epoch)
        except StopIteration:
            print("Early stopping due to no improvement.")
        self.writer.close()

    def evaluate_model(self):
        if self.test_loader is None:
            print("No test set provided.")
            return

        self.model.eval()
        test_losses, test_metrics = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                metrics = self._process_batch(batch, training=False)
                test_losses.append(metrics.pop('total_loss'))
                test_metrics.append(metrics)

        avg_test_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_test_loss:.4f}")

        if test_metrics:
            avg_metrics = {k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0]}
            for k, v in avg_metrics.items():
                print(f"  {k}: {v:.4f}")

    def save_checkpoint(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Model checkpoint saved to {filename}")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = CombinedDepthModel()
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Chuẩn hóa theo ImageNet
                             std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 8  # Tăng batch size nếu GPU của bạn cho phép
    train_dataset = DataLoadPreprocess(mode='train', transform=transform)
    test_dataset = DataLoadPreprocess(mode='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    trainer = BaseTrainer(
        model,
        train_loader,
        test_loader,
        device=device,
        num_epochs=100,
        early_stopping_patience=15,
        validation_interval=0.25  # Đánh giá mỗi 25% dữ liệu
    )
    trainer.train()
    trainer.evaluate_model()
    trainer.save_checkpoint('combined_model.pth')

if __name__ == '__main__':
    main()
