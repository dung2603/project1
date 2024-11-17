# train.py

import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.model import CombinedDepthModel
from dataloader import DataLoadPreprocess
from evaluation import compute_metrics

# Import các hàm loss từ file loss.py
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
        num_epochs=50,
        min_depth=0.1,
        max_depth=10.0,
        early_stopping_patience=10,
        initial_lr=1e-6,
        fine_tune_lr=1e-7,
        weight_decay=1e-4,
        max_grad_norm=1.0
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

        self.optimizer = self._init_optimizer(initial_lr, fine_tune_lr, weight_decay)
        self.scheduler = self._init_scheduler()

        # Sử dụng hàm loss tổng hợp từ loss.py
        self.criterion = DepthLoss(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # Thiết lập TensorBoard
        self.writer = SummaryWriter('runs/depth_estimation_experiment')

    def _init_optimizer(self, initial_lr, fine_tune_lr, weight_decay):
        """
        Khởi tạo trình tối ưu hóa AdamW với learning rate khác nhau cho backbone và phần còn lại.
        """
        fine_tune_params = [param for param in self.model.parameters() if param.requires_grad]
        optimizer = optim.AdamW(fine_tune_params, lr=initial_lr, weight_decay=weight_decay)
        return optimizer

    def _init_scheduler(self):
        """
        Khởi tạo trình điều phối học (learning rate scheduler).
        """
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

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
            self.optimizer.zero_grad()
            depth_map = self.model(images)
            depth_map = self._resize_depth_map(depth_map, depths)

            # Apply mask for valid depth values
            mask = self._mask_depth(depths)
            if mask is not None:
                depth_map = depth_map * mask
                depths = depths * mask

            # Tính toán loss sử dụng hàm loss tổng hợp
            total_loss = self.criterion(depth_map, depths)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            # Log loss to TensorBoard
            self.writer.add_scalar('Loss/Total', total_loss.item(), self.global_step)

            loss_dict = {
                'total_loss': total_loss.item()
            }

            return loss_dict
        else:
            self.model.eval()
            with torch.no_grad():
                depth_map = self.model(images)
                depth_map = self._resize_depth_map(depth_map, depths)

                # Apply mask for valid depth values
                mask = self._mask_depth(depths)
                if mask is not None:
                    depth_map = depth_map * mask
                    depths = depths * mask

                # Tính toán loss sử dụng hàm loss tổng hợp
                total_loss = self.criterion(depth_map, depths)

                # Compute metrics using compute_metrics from evaluation.py
                metrics = compute_metrics(depth_map, depths, mask=mask)

                loss_dict = {
                    'total_loss': total_loss.item()
                }
                loss_dict.update(metrics)

                # Log validation loss and metrics to TensorBoard
                self.writer.add_scalar('Validation/Loss/Total', total_loss.item(), self.global_step)
                for metric_name, metric_value in metrics.items():
                    self.writer.add_scalar(f'Validation/Metrics/{metric_name}', metric_value, self.global_step)

                return loss_dict

    def _resize_depth_map(self, depth_map, target_depths):
        """
        Điều chỉnh kích thước depth_map để khớp với depths.
        """
        if depth_map.shape != target_depths.shape:
            depth_map = F.interpolate(
                depth_map,
                size=target_depths.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        return depth_map

    def _mask_depth(self, target):
        """
        Tạo mask cho các giá trị độ sâu hợp lệ.
        """
        mask = (target > self.min_depth) & (target < self.max_depth)
        return mask.float()

    def train_on_batch(self, batch):
        """
        Huấn luyện mô hình trên một batch dữ liệu.
        """
        return self._process_batch(batch, training=True)

    def validate_on_batch(self, batch):
        """
        Đánh giá mô hình trên một batch dữ liệu.
        """
        return self._process_batch(batch, training=False)

    def validate(self):
        """
        Đánh giá mô hình trên toàn bộ bộ dữ liệu đánh giá.
        """
        if self.test_loader is None:
            print("Không có bộ dữ liệu đánh giá (test_loader) được cung cấp.")
            return

        self.model.eval()
        val_losses = []
        val_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation"):
                metrics = self.validate_on_batch(batch)
                val_losses.append(metrics.pop('total_loss'))
                val_metrics.append(metrics)

        avg_val_loss = np.mean(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if val_metrics:
            avg_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
            print("Validation Metrics:")
            for k, v in avg_metrics.items():
                print(f"  {k}: {v:.4f}")

            # Log validation metrics to TensorBoard
            for metric_name, metric_value in avg_metrics.items():
                self.writer.add_scalar(f'Validation/Averages/{metric_name}', metric_value, self.global_step)

        # Update scheduler based on validation loss
        self.scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.epochs_no_improve = 0
            self.save_checkpoint('best_model.pth')
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.epochs_no_improve} epochs with no improvement.")
                raise StopIteration

    def train_epoch(self, epoch):
        """
        Huấn luyện mô hình trong một epoch.
        """
        print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
        train_losses = []
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            self.global_step = epoch * num_batches + batch_idx
            losses = self.train_on_batch(batch)
            train_losses.append(losses['total_loss'])

            # Log training loss after each batch
            self.writer.add_scalar('Training/Loss/Total', losses['total_loss'], self.global_step)

        avg_train_loss = np.mean(train_losses)
        print(f"\nAverage Training Loss: {avg_train_loss:.4f}")

        # Validation after each epoch
        self.validate()

    def train(self):
        """
        Thực hiện huấn luyện mô hình trên toàn bộ epoch.
        """
        try:
            for epoch in range(self.num_epochs):
                self.train_epoch(epoch)
        except StopIteration:
            print("Dừng huấn luyện sớm do không cải thiện.")

        # Close TensorBoard writer
        self.writer.close()

    def evaluate_model(self):
        """
        Đánh giá mô hình trên bộ dữ liệu kiểm tra.
        """
        print("\nEvaluating on Test Set...")
        if self.test_loader is None:
            print("No test set provided.")
            return

        self.model.eval()
        test_losses = []
        test_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                metrics = self.validate_on_batch(batch)
                test_losses.append(metrics.pop('total_loss'))
                test_metrics.append(metrics)

        avg_test_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_test_loss:.4f}")

        if test_metrics:
            avg_metrics = {k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0]}
            print("Test Metrics:")
            for k, v in avg_metrics.items():
                print(f"  {k}: {v:.4f}")

    def save_checkpoint(self, filename):
        """
        Lưu trọng số mô hình vào tệp.
        """
        torch.save(self.model.state_dict(), filename)
        print(f"Model checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        """
        Tải trọng số mô hình từ tệp.
        """
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        print(f"Model checkpoint loaded from {filename}")


def main():
    # Thiết lập thiết bị
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Khởi tạo mô hình
    model = CombinedDepthModel()

    # Thiết lập Data Augmentation và Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Điều chỉnh kích thước theo nhu cầu
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Nếu sử dụng backbone pre-trained trên ImageNet
                             std=[0.229, 0.224, 0.225]),
    ])

    # Thiết lập kích thước batch
    batch_size = 4  # Điều chỉnh tùy theo hệ thống của bạn

    # Tải và tạo DataLoaders
    train_dataset = DataLoadPreprocess(mode='train', transform=transform)
    test_dataset = DataLoadPreprocess(mode='test', transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Điều chỉnh số worker tùy theo hệ thống
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Điều chỉnh số worker tùy theo hệ thống
        pin_memory=True
    )

    # Khởi tạo trainer với train_loader và test_loader
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    # Bắt đầu huấn luyện
    trainer.train()

    # Đánh giá trên bộ dữ liệu kiểm tra sau khi huấn luyện
    trainer.evaluate_model()

    # Lưu checkpoint của mô hình
    trainer.save_checkpoint('combined_model.pth')


if __name__ == '__main__':
    main()
