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
from evaluation import compute_metrics  # Updated import
# Đã loại bỏ các import không cần thiết cho loss tùy chỉnh
# from loss import CombinedLoss, SmoothnessLoss

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def compute_smoothness_loss(depth_map):
    """
    Compute smoothness loss by penalizing large gradients in the depth map.

    Args:
        depth_map (torch.Tensor): Predicted depth map of shape (B, 1, H, W)

    Returns:
        torch.Tensor: Smoothness loss
    """
    grad_x = torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:])
    grad_y = torch.abs(depth_map[:, :, :-1, :] - depth_map[:, :, 1:, :])
    smoothness_loss = torch.mean(grad_x) + torch.mean(grad_y)
    return smoothness_loss


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
        initial_lr=1e-6,  # Giảm từ 1e-5 xuống 1e-6
        fine_tune_lr=1e-7,  # Giảm từ 1e-6 xuống 1e-7
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
        self.smoothness_loss_weight = smoothness_loss_weight
        self.early_stopping_patience = early_stopping_patience
        self.max_grad_norm = max_grad_norm

        self.optimizer = self._init_optimizer(initial_lr, fine_tune_lr, weight_decay)
        self.scheduler = self._init_scheduler()

        # Thay thế CombinedLoss bằng hàm mất mát tiêu chuẩn
        self.criterion = nn.MSELoss()  # Bạn có thể thử nn.L1Loss() hoặc nn.SmoothL1Loss()

        # Không sử dụng smoothness_criterion vì chúng ta đã định nghĩa compute_smoothness_loss
        # self.smoothness_criterion = nn.L1Loss()  # Hoặc nn.MSELoss()

        # Tắt GradScaler để không sử dụng mixed precision
        self.scaler = None

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # Thiết lập TensorBoard
        self.writer = SummaryWriter('runs/depth_estimation_experiment')

    def _init_optimizer(self, initial_lr, fine_tune_lr, weight_decay):
        """
        Khởi tạo trình tối ưu hóa AdamW với learning rate khác nhau cho backbone và phần còn lại.
        """
        fine_tune_params = [param for param in self.model.parameters() if param.requires_grad]
        frozen_params = [param for param in self.model.parameters() if not param.requires_grad]

        if not fine_tune_params:
            print("Không có tham số nào yêu cầu gradient. Tất cả tham số sẽ được cập nhật với lr=1e-7.")
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
        # Sử dụng ReduceLROnPlateau để giảm learning rate khi mất mát validation không giảm
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

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

        # Kiểm tra NaN trong images và depths
        if torch.isnan(images).any():
            print("Warning: Images contain NaN values.")
        if torch.isnan(depths).any():
            print("Warning: Depths contain NaN values.")

        # In thông tin về dữ liệu đầu vào
        print(f"Images - min: {images.min().item()}, max: {images.max().item()}, mean: {images.mean().item()}")
        print(f"Depths - min: {depths.min().item()}, max: {depths.max().item()}, mean: {depths.mean().item()}")

        if training:
            self.model.train()
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=False):  # Tắt autocast
                depth_map = self.model(images)
                
                # Kiểm tra NaN trong depth_map sau khi dự đoán
                if torch.isnan(depth_map).any():
                    print("Warning: Predicted depth_map contains NaN values.")

                depth_map = self._resize_depth_map(depth_map, depths)

                # Apply mask for valid depth values
                mask = self._mask_depth(depths)
                if mask is not None:
                    depth_map = depth_map * mask
                    depths = depths * mask

                # Kiểm tra NaN trước khi tính loss
                if torch.isnan(depth_map).any() or torch.isnan(depths).any():
                    print("Warning: depth_map or depths contain NaN values before loss calculation.")

                # Sử dụng hàm mất mát tiêu chuẩn
                loss_depth = self.criterion(depth_map, depths)
                loss_smoothness = compute_smoothness_loss(depth_map)
                total_loss = loss_depth + self.smoothness_loss_weight * loss_smoothness

                # In ra giá trị loss để debug
                print(f"Training - loss_depth: {loss_depth.item()}, loss_smoothness: {loss_smoothness.item()}, total_loss: {total_loss.item()}")

            # Kiểm tra NaN trong total_loss
            if torch.isnan(total_loss):
                print("Warning: total_loss is NaN.")
                # Bạn có thể lựa chọn dừng huấn luyện hoặc tiếp tục tùy thuộc vào nhu cầu
                # raise ValueError("Total loss is NaN.")

            total_loss.backward()

            # Kiểm tra NaN trong gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"Warning: Gradient of {name} contains NaN.")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            # Log loss to TensorBoard
            self.writer.add_scalar('Loss/Depth', loss_depth.item(), self.global_step)
            self.writer.add_scalar('Loss/Smoothness', loss_smoothness.item(), self.global_step)
            self.writer.add_scalar('Loss/Total', total_loss.item(), self.global_step)

            loss_dict = {
                'depth_loss': loss_depth.item(),
                'smoothness_loss': loss_smoothness.item(),
                'total_loss': total_loss.item()
            }

            return loss_dict
        else:
            self.model.eval()
            with torch.no_grad():
                depth_map = self.model(images)
                
                # Kiểm tra NaN trong depth_map sau khi dự đoán
                if torch.isnan(depth_map).any():
                    print("Warning: Predicted depth_map contains NaN values.")

                depth_map = self._resize_depth_map(depth_map, depths)

                # Apply mask for valid depth values
                mask = self._mask_depth(depths)
                if mask is not None:
                    depth_map = depth_map * mask
                    depths = depths * mask

                # Kiểm tra NaN trước khi tính loss
                if torch.isnan(depth_map).any() or torch.isnan(depths).any():
                    print("Warning: depth_map or depths contain NaN values before loss calculation.")

                # Sử dụng hàm mất mát tiêu chuẩn
                loss_depth = self.criterion(depth_map, depths)
                loss_smoothness = compute_smoothness_loss(depth_map)
                total_loss = loss_depth + self.smoothness_loss_weight * loss_smoothness

                # In ra giá trị loss để debug
                print(f"Validation - loss_depth: {loss_depth.item()}, loss_smoothness: {loss_smoothness.item()}, total_loss: {total_loss.item()}")

                # Compute metrics using compute_metrics from metric.py
                metrics = compute_metrics(depth_map, depths, mask=mask)

                loss_dict = {
                    'depth_loss': loss_depth.item(),
                    'smoothness_loss': loss_smoothness.item(),
                    'total_loss': total_loss.item()
                }
                loss_dict.update(metrics)

                # Kiểm tra các giá trị metric có NaN không
                for k, v in metrics.items():
                    if np.isnan(v):
                        print(f"Warning: Metric {k} is NaN.")

                # Log validation loss and metrics to TensorBoard
                self.writer.add_scalar('Validation/Loss/Depth', loss_depth.item(), self.global_step)
                self.writer.add_scalar('Validation/Loss/Smoothness', loss_smoothness.item(), self.global_step)
                self.writer.add_scalar('Validation/Loss/Total', total_loss.item(), self.global_step)
                for metric_name, metric_value in metrics.items():
                    self.writer.add_scalar(f'Validation/Metrics/{metric_name}', metric_value, self.global_step)

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

    def _mask_depth(self, target):
        """
        Tạo mask cho các giá trị độ sâu hợp lệ.

        :param target: Bản đồ độ sâu thực tế.
        :return: Mask với giá trị 1 cho các điểm hợp lệ và 0 cho các điểm không hợp lệ.
        """
        mask = (target > self.min_depth) & (target < self.max_depth)
        # Thêm kiểm tra tỷ lệ hợp lệ
        valid_ratio = mask.float().mean().item()
        print(f"Valid depth ratio: {valid_ratio:.4f}")
        return mask.float()

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
            self.global_step = epoch * num_batches + batch_idx
            losses = self.train_on_batch(batch)
            train_losses.append(losses['total_loss'])

            # Log training loss after each batch
            self.writer.add_scalar('Training/Loss/Total', losses['total_loss'], self.global_step)

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

        :param filename: Tên tệp để lưu trọng số.
        """
        # Kiểm tra NaN trong trọng số mô hình trước khi lưu
        has_nan = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"Warning: Parameter {name} contains NaN values.")
                has_nan = True
        if not has_nan:
            torch.save(self.model.state_dict(), filename)
            print(f"Model checkpoint saved to {filename}")
        else:
            print("Checkpoint not saved due to NaN values in model parameters.")

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

    # Kiểm tra NaN trong trọng số mô hình ngay sau khi khởi tạo
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Warning: Parameter {name} contains NaN values at initialization.")

    # **Mở khóa các tham số cần thiết**
    # Ví dụ: Mở khóa các lớp của Segmentation Backbone
    # model.unfreeze_backbone_layers(layers_to_unfreeze=['layer4', 'block'])

    # Thiết lập Data Augmentation và Transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((256, 512)),  # Ví dụ kích thước, điều chỉnh theo nhu cầu
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

    # Kiểm tra dữ liệu đầu vào trước khi huấn luyện
    sample_batch = next(iter(train_loader))
    images_sample = sample_batch['image'].to(device)
    depths_sample = sample_batch['depth'].to(device)
    masks_sample = sample_batch.get('mask', None)
    if masks_sample is not None:
        masks_sample = masks_sample.to(device)

    print(f"Sample Images - min: {images_sample.min().item()}, max: {images_sample.max().item()}, mean: {images_sample.mean().item()}")
    print(f"Sample Depths - min: {depths_sample.min().item()}, max: {depths_sample.max().item()}, mean: {depths_sample.mean().item()}")

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
