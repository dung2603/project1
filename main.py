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
from evaluation import compute_all_metrics, mask_depth
from loss import CombinedLoss, SmoothnessLoss


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
        smoothness_loss_weight=0.1,
        early_stopping_patience=10,
        alpha=0.5,
        beta=0.1,
        initial_lr=1e-4,
        fine_tune_lr=1e-5,
        weight_decay=1e-4,
        max_grad_norm=1.0
    ):
        """
        Khởi tạo lớp BaseTrainer với mô hình, DataLoader huấn luyện và đánh giá.

        :param model: Mô hình PyTorch được huấn luyện.
        :param train_loader: DataLoader cho tập huấn luyện.
        :param test_loader: DataLoader cho tập đánh giá.
        :param device: Thiết bị để huấn luyện (CPU hoặc GPU).
        :param num_epochs: Số epoch để huấn luyện.
        :param min_depth: Độ sâu tối thiểu.
        :param max_depth: Độ sâu tối đa.
        :param smoothness_loss_weight: Trọng số cho SmoothnessLoss.
        :param early_stopping_patience: Số epoch không cải thiện để dừng sớm.
        :param alpha: Trọng số alpha cho CombinedLoss.
        :param beta: Trọng số beta cho CombinedLoss.
        :param initial_lr: Learning rate cho các tham số cần fine-tune.
        :param fine_tune_lr: Learning rate cho các tham số không cần fine-tune.
        :param weight_decay: Hệ số weight decay cho optimizer.
        :param max_grad_norm: Giới hạn norm của gradient để clip.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.smoothness_loss_weight = smoothness_loss_weight
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm

        self.optimizer = self._init_optimizer(initial_lr, fine_tune_lr, weight_decay)
        self.scheduler = self._init_scheduler()
        self.criterion = CombinedLoss(alpha=alpha, beta=beta)
        self.smoothness_criterion = SmoothnessLoss()
        self.scaler = torch.cuda.amp.GradScaler()

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def _init_optimizer(self, initial_lr, fine_tune_lr, weight_decay):
        """
        Khởi tạo trình tối ưu hóa AdamW với learning rate khác nhau cho backbone và phần còn lại.
        """
        fine_tune_params = [param for param in self.model.parameters() if param.requires_grad]
        frozen_params = [param for param in self.model.parameters() if not param.requires_grad]

        if not fine_tune_params:
            print("Không có tham số nào yêu cầu gradient. Tất cả tham số sẽ được cập nhật với lr=1e-5.")
            optimizer = optim.AdamW(frozen_params, lr=fine_tune_lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW([
                {'params': fine_tune_params, 'lr': initial_lr},
                {'params': frozen_params, 'lr': fine_tune_lr}
            ], weight_decay=weight_decay)

        return optimizer

    def _init_scheduler(self):
        """
        Khởi tạo trình điều phối học (learning rate scheduler).
        """
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def _process_batch(self, batch, training=True):
        """
        Xử lý một batch dữ liệu cho huấn luyện hoặc đánh giá.

        :param batch: Một batch dữ liệu từ DataLoader.
        :param training: Boolean xác định chế độ huấn luyện hay đánh giá.
        :return: Từ điển chứa các giá trị loss và metrics nếu đánh giá.
        """
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
            with torch.cuda.amp.autocast():
                depth_map = self.model(images)
                depth_map = self._resize_depth_map(depth_map, depths)

                mask = mask_depth(target=depths, min_depth=self.min_depth, max_depth=self.max_depth)
                if mask is not None:
                    depth_map = depth_map * mask
                    depths = depths * mask

                loss_depth_ssim = self.criterion(depth_map, depths)
                loss_smoothness = self.smoothness_criterion(depth_map)
                total_loss = loss_depth_ssim + self.smoothness_loss_weight * loss_smoothness

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_dict = {
                'depth_loss_ssim': loss_depth_ssim.item(),
                'smoothness_loss': loss_smoothness.item(),
                'total_loss': total_loss.item()
            }

            return loss_dict
        else:
            self.model.eval()
            with torch.no_grad():
                depth_map = self.model(images)
                depth_map = self._resize_depth_map(depth_map, depths)

                mask = mask_depth(target=depths, min_depth=self.min_depth, max_depth=self.max_depth)
                if mask is not None:
                    depth_map = depth_map * mask
                    depths = depths * mask

                loss_depth_ssim = self.criterion(depth_map, depths)
                loss_smoothness = self.smoothness_criterion(depth_map)
                total_loss = loss_depth_ssim + self.smoothness_loss_weight * loss_smoothness

                metrics = compute_all_metrics(depth_map, depths, mask=mask)

                loss_dict = {
                    'depth_loss_ssim': loss_depth_ssim.item(),
                    'smoothness_loss': loss_smoothness.item(),
                    'total_loss': total_loss.item()
                }
                loss_dict.update(metrics)

                return loss_dict

    def _resize_depth_map(self, depth_map, target_depths):
        """
        Điều chỉnh kích thước depth_map để khớp với depths.

        :param depth_map: Bản đồ độ sâu dự đoán.
        :param target_depths: Bản đồ độ sâu thực tế.
        :return: Depth map đã được điều chỉnh kích thước.
        """
        if depth_map.shape != target_depths.shape:
            depth_map = F.interpolate(
                depth_map,
                size=target_depths.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        return depth_map

    def train_on_batch(self, batch):
        """
        Huấn luyện mô hình trên một batch dữ liệu.

        :param batch: Một batch dữ liệu từ DataLoader.
        :return: Từ điển chứa các giá trị loss.
        """
        return self._process_batch(batch, training=True)

    def validate_on_batch(self, batch):
        """
        Đánh giá mô hình trên một batch dữ liệu.

        :param batch: Một batch dữ liệu từ DataLoader đánh giá.
        :return: Từ điển chứa các giá trị loss và metrics.
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

    def train_epoch(self, epoch):
        """
        Huấn luyện mô hình trong một epoch.

        :param epoch: Số thứ tự của epoch hiện tại.
        """
        print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
        train_losses = []
        num_batches = len(self.train_loader)

        validation_steps = [
            int(0.25 * num_batches),
            int(0.5 * num_batches),
            int(0.75 * num_batches),
            num_batches
        ]
        current_val_step = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            losses = self.train_on_batch(batch)
            train_losses.append(losses['total_loss'])

            if (batch_idx + 1) == validation_steps[current_val_step]:
                print(f"\nValidation after {int((current_val_step + 1) * 25)}% of the epoch")
                self.validate()
                current_val_step += 1
                if current_val_step >= len(validation_steps):
                    break

        avg_train_loss = np.mean(train_losses)
        print(f"\nAverage Training Loss: {avg_train_loss:.4f}")
        print("\nValidation after complete epoch")
        self.validate()

        self.scheduler.step()

        # Early Stopping
        if avg_train_loss < self.best_val_loss:
            self.best_val_loss = avg_train_loss
            self.epochs_no_improve = 0
            self.save_checkpoint('best_model.pth')
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.epochs_no_improve} epochs with no improvement.")
                raise StopIteration

    def train(self):
        """
        Thực hiện huấn luyện mô hình trên toàn bộ epoch.
        Thực hiện đánh giá sau mỗi 25% của epoch và sau khi hoàn thành epoch.
        """
        try:
            for epoch in range(self.num_epochs):
                self.train_epoch(epoch)
        except StopIteration:
            print("Dừng huấn luyện sớm do không cải thiện.")

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

        :param filename: Tên tệp để lưu trọng số.
        """
        torch.save(self.model.state_dict(), filename)
        print(f"Model checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        """
        Tải trọng số mô hình từ tệp.

        :param filename: Tên tệp chứa trọng số.
        """
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        print(f"Model checkpoint loaded from {filename}")


def main():
    # Thiết lập thiết bị
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Khởi tạo mô hình
    model = CombinedDepthModel()

    # **Mở khóa các tham số cần thiết**
    # Ví dụ: Mở khóa các lớp của Segmentation Backbone
    # model.unfreeze_backbone_layers(layers_to_unfreeze=['layer4', 'block'])

    # Thiết lập kích thước batch
    batch_size = 2  # Điều chỉnh tùy theo hệ thống của bạn

    # Tải và tạo DataLoaders
    train_dataset = DataLoadPreprocess(mode='train')
    test_dataset = DataLoadPreprocess(mode='test')

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

    # **Mở khóa các tham số nếu cần**
    # trainer.model.unfreeze_backbone_layers(layers_to_unfreeze=['layer4', 'block'])

    # Bắt đầu huấn luyện
    trainer.train()

    # Đánh giá trên bộ dữ liệu kiểm tra sau khi huấn luyện
    trainer.evaluate_model()

    # Lưu checkpoint của mô hình
    trainer.save_checkpoint('combined_model.pth')


if __name__ == '__main__':
    main()
