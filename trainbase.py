# train.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

from model.model import CombinedModel
from dataloader import DataLoadPreprocess
from evaluation import compute_metrics
from loss import silog_loss, ssim_loss, gradl1_loss


class BaseTrainer:
    def __init__(self, model, train_loader, test_loader=None, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.depth_criterion = nn.L1Loss()
        self.seg_criterion = nn.CrossEntropyLoss()
        self.num_epochs = 10
        self.min_depth = 0.1
        self.max_depth = 10.0
        self.scaler = torch.cuda.amp.GradScaler()

        # Additional loss weights
        self.silog_loss_weight = 0.1
        self.ssim_loss_weight = 0.1
        self.gradl1_loss_weight = 0.1

    def init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

    def init_scheduler(self):
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def train_on_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        images = batch['image'].to(self.device)
        depths = batch['depth'].to(self.device)
        masks = batch['mask'].to(self.device)

        if depths.ndim == 3:
            depths = depths.unsqueeze(1)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        with torch.cuda.amp.autocast():
            outputs, seg_output = self.model(images)
            if outputs.shape[2:] != depths.shape[2:]:
                outputs = F.interpolate(outputs, size=depths.shape[2:], mode='bilinear', align_corners=True)

            # Apply mask
            outputs = outputs * masks
            depths = depths * masks

            # Compute losses
            l1_loss = self.depth_criterion(outputs, depths)
            silog_loss_value = silog_loss(outputs, depths, mask=masks)
            ssim_loss_value = ssim_loss(outputs / self.max_depth, depths / self.max_depth)
            gradl1_loss_value = gradl1_loss(outputs, depths)

            depth_loss = l1_loss + \
                         self.silog_loss_weight * silog_loss_value + \
                         self.ssim_loss_weight * ssim_loss_value + \
                         self.gradl1_loss_weight * gradl1_loss_value

            if 'segmentation' in batch:
                segmentations = batch['segmentation'].to(self.device)
                seg_loss = self.seg_criterion(seg_output, segmentations)
                total_loss = depth_loss + seg_loss
            else:
                total_loss = depth_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # Return losses for logging
        loss_dict = {
            'l1_loss': l1_loss.item(),
            'silog_loss': silog_loss_value.item(),
            'ssim_loss': ssim_loss_value.item(),
            'gradl1_loss': gradl1_loss_value.item(),
            'depth_loss': depth_loss.item()
        }
        if 'segmentation' in batch:
            loss_dict['seg_loss'] = seg_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        return loss_dict

    def validate_on_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            masks = batch['mask'].to(self.device)

            if depths.ndim == 3:
                depths = depths.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            if 'segmentation' in batch:
                segmentations = batch['segmentation'].to(self.device)
            else:
                segmentations = None

            outputs, seg_output = self.model(images)
            if outputs.shape[2:] != depths.shape[2:]:
                outputs = F.interpolate(outputs, size=depths.shape[2:], mode='bilinear', align_corners=True)

            # Apply mask
            outputs = outputs * masks
            depths = depths * masks

            # Compute losses
            l1_loss = self.depth_criterion(outputs, depths)
            silog_loss_value = silog_loss(outputs, depths, mask=masks)
            ssim_loss_value = ssim_loss(outputs / self.max_depth, depths / self.max_depth)
            gradl1_loss_value = gradl1_loss(outputs, depths)

            depth_loss = l1_loss + \
                         self.silog_loss_weight * silog_loss_value + \
                         self.ssim_loss_weight * ssim_loss_value + \
                         self.gradl1_loss_weight * gradl1_loss_value

            if segmentations is not None:
                seg_loss = self.seg_criterion(seg_output, segmentations)
                total_loss = depth_loss + seg_loss
            else:
                total_loss = depth_loss

            # Compute evaluation metrics
            metrics = compute_metrics(depths, outputs,
                                      min_depth=self.min_depth,
                                      max_depth=self.max_depth)

            # Return losses and metrics
            loss_dict = {
                'l1_loss': l1_loss.item(),
                'silog_loss': silog_loss_value.item(),
                'ssim_loss': ssim_loss_value.item(),
                'gradl1_loss': gradl1_loss_value.item(),
                'depth_loss': depth_loss.item()
            }
            if segmentations is not None:
                loss_dict['seg_loss'] = seg_loss.item()
            loss_dict['total_loss'] = total_loss.item()

            return {**loss_dict, **metrics}

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            train_losses = []
            self.model.train()
            num_batches = len(self.train_loader)
            validation_intervals = max(1, num_batches // 4)
            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                losses = self.train_on_batch(batch)
                train_losses.append(losses['total_loss'])

                if (batch_idx + 1) % validation_intervals == 0:
                    print(f"Validation at {(batch_idx + 1) / num_batches * 100:.2f}% of epoch")
                    if self.test_loader is not None:
                        val_losses = []
                        val_metrics = []
                        self.model.eval()
                        for val_batch in self.test_loader:
                            metrics = self.validate_on_batch(val_batch)
                            val_losses.append(metrics['total_loss'])
                            val_metrics.append({k: metrics[k] for k in metrics if k not in ['total_loss', 'depth_loss', 'seg_loss', 'l1_loss', 'silog_loss', 'ssim_loss', 'gradl1_loss']})

                        avg_val_loss = np.mean(val_losses)
                        avg_metrics = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0].keys()}
                        print(f"Validation Loss: {avg_val_loss}")
                        print("Validation Metrics:")
                        for k, v in avg_metrics.items():
                            print(f"  {k}: {v}")

            avg_train_loss = np.mean(train_losses)
            print(f"Average Training Loss: {avg_train_loss}")

            self.scheduler.step()

    def save_checkpoint(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_checkpoint(self, filename):
        self.model.load_state_dict(torch.load(filename))


def main():
    # Set up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Instantiate the model
    n_classes = 40  # Adjust according to your number of segmentation classes
    model = CombinedModel(n_classes=n_classes).to(device)
    print(model)

    # Set batch size
    batch_size = 16  # Adjust batch size according to your needs and system capacity

    # Create data loaders with batch size
    train_dataset = DataLoadPreprocess(mode='train')
    test_dataset = DataLoadPreprocess(mode='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust as needed
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Adjust as needed
        pin_memory=True
    )

    # Instantiate the trainer
    trainer = BaseTrainer(model, train_loader, test_loader, device=device)

    # Start training
    trainer.train()

    # Save the model
    trainer.save_checkpoint('combined_model.pth')


if __name__ == '__main__':
    main()
