"""
Training entrypoint and utilities for pose canonicalization with MM-Fi dataset.
Includes a CLI that loads YAML configs and supports a --sanity-check mode.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional
import logging
from tqdm import tqdm
import os
import logging
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
from models.model import create_pose_canonicalization_model, PoseCanonicalizationNet
from data_loader import create_data_loaders
from models.losses import PoseCanonicalizationLoss
from utils.config_utils import ConfigManager
from evaluate import evaluate_pose_canonicalization, compute_similarity_transform


class PCTrainer:
    """
    Trainer for pose canonicalization model with MM-Fi dataset
    """
    def __init__(self, 
                 model: PoseCanonicalizationNet,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 pose_weight: float = 1.0,
                 rotation_weight: float = 1.0,
                 weight_decay: float = 1e-4,
                 scheduler_step_size: int = 30,
                 scheduler_gamma: float = 0.1,
                 log_interval: int = 10):
        """
        Initialize trainer
        Args:
            model: PoseCanonicalizationNet model
            device: Training device
            learning_rate: Learning rate for optimizer
            pose_weight: Weight for pose reconstruction loss
            rotation_weight: Weight for rotation loss
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = PoseCanonicalizationLoss(pose_weight=pose_weight, rotation_weight=rotation_weight)
        
        # Optimizer (following 6DRepNet training setup)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=scheduler_step_size, 
            gamma=scheduler_gamma
        )
        self.logger = logging.getLogger(__name__)
        self.log_interval = max(1, int(log_interval))
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step using pre-generated training pairs from data loader
        
        Args:
            batch: Batch from MMFiCanonPose data loader containing:
                - 'input_pose': (batch, 17, 3) rotated poses (network input)
                - 'canonical_pose': (batch, 17, 3) canonical poses (ground truth)
                - 'rotation_matrix': (batch, 3, 3) rotation matrices (ground truth)
                
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        # Move batch to device
        input_poses = batch['input_pose'].to(self.device)
        target_canonical = batch['canonical_pose'].to(self.device)
        target_rotations = batch['rotation_matrix'].to(self.device)
        # Forward pass
        pred_canonical, pred_rotation_repr = self.model(input_poses)
        pred_rotation_matrix = self.model.get_rotation_matrix(pred_rotation_repr)
        # Compute loss with cycle and residual (if applicable)
        residual = None
        if hasattr(self.model, 'predict_mode') and self.model.predict_mode == 'rotation_plus_residual':
            R_inv = pred_rotation_matrix.transpose(1, 2)
            canon_by_rot = torch.bmm(input_poses.reshape(input_poses.size(0), -1, 3), R_inv).reshape_as(pred_canonical)
            residual = pred_canonical - canon_by_rot
        loss_dict = self.criterion(
            pred_canonical, pred_rotation_matrix,
            target_canonical, target_rotations,
            input_pose=input_poses,
            residual=residual
        )
        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        self.optimizer.step()
        
        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single validation step using pre-generated training pairs from data loader
        Args:
            batch: Batch from MMFiCanonPose data loader
        Returns:
            Dictionary with loss values and metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            input_poses = batch['input_pose'].to(self.device)
            target_canonical = batch['canonical_pose'].to(self.device)
            target_rotations = batch['rotation_matrix'].to(self.device)
            # Forward pass
            pred_canonical, pred_rotation_repr = self.model(input_poses)
            pred_rotation_matrix = self.model.get_rotation_matrix(pred_rotation_repr)
            # Compute loss with cycle and residual (if applicable)
            residual = None
            if hasattr(self.model, 'predict_mode') and self.model.predict_mode == 'rotation_plus_residual':
                R_inv = pred_rotation_matrix.transpose(1, 2)
                canon_by_rot = torch.bmm(input_poses.reshape(input_poses.size(0), -1, 3), R_inv).reshape_as(pred_canonical)
                residual = pred_canonical - canon_by_rot
            loss_dict = self.criterion(
                pred_canonical, pred_rotation_matrix,
                target_canonical, target_rotations,
                input_pose=input_poses,
                residual=residual
            )
            # Additional metrics
            pose_error = torch.mean(torch.norm(pred_canonical - target_canonical, dim=-1))
            # Rotation error in degrees
            rotation_error_rad = loss_dict['rotation_loss']
            rotation_error_deg = rotation_error_rad * 180.0 / np.pi
            
            # Compute similarity transform metrics
            eval_metrics = evaluate_pose_canonicalization(
                pred_canonical, target_canonical, pred_rotation_matrix, target_rotations
            )
            
            loss_dict.update({
                'pose_error_mm': pose_error.item() * 1000,  # Convert to mm
                'rotation_error_deg': rotation_error_deg,
                'mpjpe': eval_metrics['mpjpe'],
                'pampjpe': eval_metrics['pampjpe']
            })
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = None, total_epochs: int = None) -> Dict[str, float]:
        """
        Train for one epoch
        Args:
            train_loader: DataLoader from create_data_loaders()
            epoch: Current epoch number (for display)
            total_epochs: Total number of epochs (for display)
        Returns:
            Dictionary with average losses for the epoch
        """
        total_losses = {}
        num_batches = 0
        
        # Create epoch description for progress bar
        epoch_desc = f"Training"
        if epoch is not None and total_epochs is not None:
            epoch_desc = f"Epoch {epoch}/{total_epochs}: Training"
        
        progress_bar = tqdm(train_loader, desc=epoch_desc)
        
        for batch in progress_bar:
            # Training step with pre-generated pairs
            losses = self.train_step(batch)
            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] = total_losses.get(key, 0) + value
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': f"{losses['total_loss']:.4f}",
                'pose_loss': f"{losses['pose_loss']:.4f}",
                'rot_loss': f"{losses['rotation_loss']:.4f}"
            })
            if num_batches % self.log_interval == 0:
                self.logger.info(
                    f"Batch {num_batches}: total={losses['total_loss']:.4f} "
                    f"pose={losses['pose_loss']:.4f} rot={losses['rotation_loss']:.4f}"
                )
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int = None, total_epochs: int = None) -> Dict[str, float]:
        """
        Validate for one epoch
        Args:
            val_loader: Validation DataLoader from create_data_loaders()
            epoch: Current epoch number (for display)
            total_epochs: Total number of epochs (for display)
        Returns:
            Dictionary with average losses for the epoch
        """
        total_losses = {}
        num_batches = 0
        
        # Create epoch description for progress bar
        epoch_desc = f"Validation"
        if epoch is not None and total_epochs is not None:
            epoch_desc = f"Epoch {epoch}/{total_epochs}: Validation"
        
        progress_bar = tqdm(val_loader, desc=epoch_desc)
        
        for batch in progress_bar:
            # Validation step with pre-generated pairs
            losses = self.validate_step(batch)
            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] = total_losses.get(key, 0) + value
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': f"{losses['total_loss']:.4f}",
                'pose_err': f"{losses['pose_error_mm']:.1f}mm",
                'rot_err': f"{losses['rotation_error_deg']:.1f}°",
                'mpjpe': f"{losses['mpjpe']:.3f}",
                'pampjpe': f"{losses['pampjpe']:.3f}"
            })
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              num_epochs: int = 100,
              save_dir: str = "./checkpoints",
              save_best: bool = True,
              save_last: bool = True,
              save_interval: int = 10) -> Dict[str, list]:
        """
        Full training loop
        Args:
            train_loader: Training DataLoader from create_data_loaders()
            val_loader: Validation DataLoader from create_data_loaders()
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
        Returns:
            Dictionary with training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        history = {
            'train_total_loss': [],
            'train_pose_loss': [],
            'train_rotation_loss': [],
            'val_total_loss': [],
            'val_pose_loss': [],
            'val_rotation_loss': [],
            'val_pose_error_mm': [],
            'val_rotation_error_deg': [],
            'val_mpjpe': [],
            'val_pampjpe': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            # Training
            train_losses = self.train_epoch(train_loader, epoch + 1, num_epochs)
            # Validation
            val_losses = self.validate_epoch(val_loader, epoch + 1, num_epochs)
            # Learning rate scheduling
            self.scheduler.step()
            # Update history
            history['train_total_loss'].append(train_losses['total_loss'])
            history['train_pose_loss'].append(train_losses.get('pose_loss', float('nan')))
            history['train_rotation_loss'].append(train_losses.get('rotation_loss', float('nan')))
            history['val_total_loss'].append(val_losses['total_loss'])
            history['val_pose_loss'].append(val_losses.get('pose_loss', float('nan')))
            history['val_rotation_loss'].append(val_losses.get('rotation_loss', float('nan')))
            history['val_pose_error_mm'].append(val_losses['pose_error_mm'])
            history['val_rotation_error_deg'].append(val_losses['rotation_error_deg'])
            history['val_mpjpe'].append(val_losses['mpjpe'])
            history['val_pampjpe'].append(val_losses['pampjpe'])
            # Save best model
            if save_best and val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'history': history
                }, os.path.join(save_dir, 'best_model.pth'))
            
            # Regular checkpoint
            if save_interval and (epoch + 1) % max(1, int(save_interval)) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': history
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
            # Logging
            self.logger.info(
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Val Loss: {val_losses['total_loss']:.4f}, "
                f"Pose Error: {val_losses['pose_error_mm']:.1f}mm, "
                f"Rotation Error: {val_losses['rotation_error_deg']:.1f}°, "
                f"MPJPE: {val_losses['mpjpe']:.3f}, "
                f"PA-MPJPE: {val_losses['pampjpe']:.3f}"
            )
        # Save last
        if save_last:
            torch.save({
                'epoch': num_epochs - 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': history
            }, os.path.join(save_dir, 'last_model.pth'))
        
        # Plot training curves
        try:
            epochs = list(range(1, len(history['train_total_loss']) + 1))
            plt.figure(figsize=(12, 10))
            # Total loss
            plt.subplot(3, 2, 1)
            plt.plot(epochs, history['train_total_loss'], label='train')
            plt.plot(epochs, history['val_total_loss'], label='val')
            plt.title('Total Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            # Pose loss
            plt.subplot(3, 2, 2)
            plt.plot(epochs, history['train_pose_loss'], label='train')
            plt.plot(epochs, history['val_pose_loss'], label='val')
            plt.title('Pose Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            # Rotation loss
            plt.subplot(3, 2, 3)
            plt.plot(epochs, history['train_rotation_loss'], label='train')
            plt.plot(epochs, history['val_rotation_loss'], label='val')
            plt.title('Rotation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            # Pose error
            plt.subplot(3, 2, 4)
            plt.plot(epochs, history['val_pose_error_mm'], label='val')
            plt.title('Pose Error (mm)'); plt.xlabel('Epoch'); plt.ylabel('mm')
            # MPJPE
            plt.subplot(3, 2, 5)
            plt.plot(epochs, history['val_mpjpe'], label='val')
            plt.title('MPJPE'); plt.xlabel('Epoch'); plt.ylabel('Error')
            # Rotation error
            plt.subplot(3, 2, 6)
            plt.plot(epochs, history['val_rotation_error_deg'], label='val')
            plt.title('Rotation Error (deg)'); plt.xlabel('Epoch'); plt.ylabel('deg')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_curves.png'))
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to plot training curves: {e}")
        
        return history
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model checkpoint
        Args:
            checkpoint_path: Path to checkpoint file
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def save_checkpoint(self, epoch: int, save_path: str, additional_info: Optional[Dict] = None):
        """
        Save model checkpoint
        Args:
            epoch: Current epoch
            save_path: Path to save checkpoint
            additional_info: Additional information to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set
        Args:
            test_loader: Test DataLoader from create_data_loaders()
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_metrics = {
            "pose_error_mm": 0.0, 
            "rotation_error": 0.0,
            "mpjpe": 0.0,
            "pampjpe": 0.0
        }
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch_size = batch['input_pose'].shape[0]
                
                # Move batch to device
                input_poses = batch['input_pose'].to(self.device)
                target_canonical = batch['canonical_pose'].to(self.device)
                target_rotations = batch['rotation_matrix'].to(self.device)
                
                # Forward pass
                pred_canonical, pred_rotation_repr = self.model(input_poses)
                pred_rotation_matrix = self.model.get_rotation_matrix(pred_rotation_repr)
                
                # Compute metrics
                pose_error = torch.mean(torch.norm(pred_canonical - target_canonical, dim=-1), dim=1)  # (batch,)
                rotation_error = torch.norm(pred_rotation_matrix - target_rotations, dim=(1, 2))  # (batch,)
                
                # Compute per-sample MPJPE and PA-MPJPE for proper accumulation
                batch_mpjpe = []
                batch_pampjpe = []
                
                for i in range(batch_size):
                    # Per-sample pose error (MPJPE)
                    sample_mpjpe = torch.mean(torch.norm(pred_canonical[i] - target_canonical[i], dim=-1))
                    batch_mpjpe.append(sample_mpjpe.item())
                    
                    # Per-sample PA-MPJPE using similarity transform
                    sample_pred = pred_canonical[i].cpu().numpy()  # (17, 3) - correct shape
                    sample_gt = target_canonical[i].cpu().numpy()   # (17, 3) - correct shape
                    _, Z, T, b, c = compute_similarity_transform(sample_gt, sample_pred, compute_optimal_scale=True)
                    sample_pred_aligned = (b * sample_pred.dot(T)) + c
                    sample_pampjpe = np.mean(np.sqrt(np.sum(np.square(sample_pred_aligned - sample_gt), axis=1)))
                    batch_pampjpe.append(sample_pampjpe)
                
                # Accumulate metrics
                total_metrics['pose_error_mm'] += torch.sum(pose_error).item() * 1000.0
                total_metrics['rotation_error'] += torch.sum(rotation_error).item()
                total_metrics['mpjpe'] += sum(batch_mpjpe)
                total_metrics['pampjpe'] += sum(batch_pampjpe)
                
                num_samples += batch_size
        
        # Average metrics
        avg_metrics = {key: (value / max(1, num_samples)) for key, value in total_metrics.items()}
        return avg_metrics


if __name__ == "__main__":
    import argparse
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train 3DPCNet Pose Canonicalization")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config. Defaults to base config if omitted.")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="Overrides as dot.key=value pairs")
    parser.add_argument("--sanity-check", action="store_true", help="Run a quick synthetic forward/backward check and exit")
    args = parser.parse_args()

    cfg_mgr = ConfigManager(args.config) if args.config else ConfigManager()
    for ov in args.override:
        if "=" not in ov:
            logger.warning(f"Ignoring invalid override: {ov}")
            continue
        key, value = ov.split("=", 1)
        if value.lower() in ["true", "false"]:
            parsed = value.lower() == "true"
        else:
            try:
                parsed = float(value) if ("." in value or value.isdigit()) else value
                if isinstance(parsed, float) and parsed.is_integer():
                    parsed = int(parsed)
            except Exception:
                parsed = value
        cfg_mgr.set(key, parsed)
    config = cfg_mgr.to_dict()

    # Configure logging from config + file handler
    log_cfg = cfg_mgr.get('logging', {}) if 'cfg_mgr' in locals() else {}
    log_level_name = log_cfg.get('log_level', 'INFO') if isinstance(log_cfg, dict) else 'INFO'
    level = getattr(logging, str(log_level_name).upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    # Prepare timestamped run directory under checkpoints
    base_ckpt_dir = config.get('checkpoints', {}).get('save_dir', './checkpoints')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_ckpt_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    # File handler writes everything printed to logger
    file_handler = logging.FileHandler(os.path.join(run_dir, 'training.log'))
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    # Avoid duplicate handlers if rerun
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == file_handler.baseFilename for h in root_logger.handlers):
        root_logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Build model/trainer
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    ckpt_cfg = config.get('checkpoints', {})

    model = create_pose_canonicalization_model(
        input_joints=model_cfg.get('input_joints', 17),
        encoder_type=model_cfg.get('encoder_type', 'mlp'),
        rotation_type=model_cfg.get('rotation_type', '6d'),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        encoder_output_dim=model_cfg.get('encoder_output_dim', 256),
        dropout=model_cfg.get('dropout', 0.1),
        predict_mode=model_cfg.get('predict_mode', 'rotation_only')
    )

    loss_cfg = config.get('loss', {})
    trainer = PCTrainer(
        model=model,
        device=device,
        learning_rate=float(train_cfg.get('learning_rate', 1e-4)),
        pose_weight=float(loss_cfg.get('pose_weight', 1.0)),
        rotation_weight=float(loss_cfg.get('rotation_weight', 1.0)),
        weight_decay=float(train_cfg.get('weight_decay', 1e-4)),
        scheduler_step_size=int(train_cfg.get('scheduler_step_size', 30)),
        scheduler_gamma=float(train_cfg.get('scheduler_gamma', 0.1)),
        log_interval=int(cfg_mgr.get('logging.log_interval', 10))
    )

    logger.info("Trainer created successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.sanity_check:
        logger.info("Running sanity check...")
        model.train()
        batch_size = 4
        joints = model_cfg.get('input_joints', 17)
        canonical_pose = torch.randn(batch_size, joints, 3, device=device)
        pred_canonical, rotation_repr = model(canonical_pose)
        pred_rotation = model.get_rotation_matrix(rotation_repr)
        criterion = PoseCanonicalizationLoss(
            pose_weight=float(loss_cfg.get('pose_weight', 1.0)),
            rotation_weight=float(loss_cfg.get('rotation_weight', 1.0)),
            cycle_weight=float(loss_cfg.get('cycle_weight', 0.0)),
            residual_l2_weight=float(loss_cfg.get('residual_l2_weight', 0.0)),
            orthogonality_weight=float(loss_cfg.get('orthogonality_weight', 0.0)),
            perceptual_weight=float(loss_cfg.get('perceptual_weight', 0.0))
        )
        losses = criterion(pred_canonical, pred_rotation, canonical_pose, pred_rotation.detach())
        loss = losses['total_loss']
        loss.backward()
        logger.info(f"Sanity check OK. Loss: {loss.item():.4f}")
        raise SystemExit(0)

    # Data loaders
    data_root = data_cfg.get('train_data_path', '') or data_cfg.get('data_root', '')
    if not data_root or not os.path.exists(data_root):
        logger.warning("Dataset path not set or not found. Set data.train_data_path in config.")
        raise SystemExit(1)

    # Axis remap config
    axis_cfg_dict = data_cfg.get('axis_remap', {})
    from data_loader import AxisRemapConfig, SplitConfig
    axis_cfg = AxisRemapConfig(
        enabled=bool(axis_cfg_dict.get('enabled', True)),
        order=tuple(axis_cfg_dict.get('new_from_old', [2, 0, 1])),
        flip=tuple(axis_cfg_dict.get('flip', [1, -1, -1]))
    )
    split_cfg_dict = data_cfg.get('split', {})
    split_cfg = SplitConfig(
        setting=split_cfg_dict.get('setting', None),
        seed=int(split_cfg_dict.get('seed', 42)),
        s1_train_ratio=float(split_cfg_dict.get('s1_train_ratio', 0.75)),
        val_ratio_from_train=float(split_cfg_dict.get('s1_val_ratio_from_train', 0.1)),
        s2_train_subjects=split_cfg_dict.get('s2_train_subjects', None),
        s2_test_subjects=split_cfg_dict.get('s2_test_subjects', None),
        s3_train_envs=split_cfg_dict.get('s3_train_envs', None),
        s3_test_envs=split_cfg_dict.get('s3_test_envs', None),
        load_from_path=bool(split_cfg_dict.get('load_from_path', False)),
        load_dir=split_cfg_dict.get('load_dir', None),
        save_dir=split_cfg_dict.get('save_dir', None)
    )

    # Determine environments to scan strictly from config format
    # - S3: union of s3_train_envs and s3_test_envs
    # - S1/S2: discover all (None)
    # - Default: discover all
    environments_to_scan = None
    if split_cfg.setting == 'S3':
        train_envs = split_cfg.s3_train_envs or []
        test_envs = split_cfg.s3_test_envs or []
        union_envs = sorted(list({*train_envs, *test_envs}))
        environments_to_scan = union_envs if len(union_envs) > 0 else None
    elif split_cfg.setting in {'S1', 'S2'}:
        environments_to_scan = None
    logging.getLogger(__name__).info(f"Environments to scan: {environments_to_scan if environments_to_scan is not None else 'ALL (discover)'}")
    if split_cfg.load_from_path and split_cfg.load_dir:
        logging.getLogger(__name__).info(f"Using precomputed datasets from: {split_cfg.load_dir}")
    if (not split_cfg.load_from_path) and split_cfg.save_dir:
        logging.getLogger(__name__).info(f"Will save materialized datasets to: {split_cfg.save_dir}")

    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=data_root,
        batch_size=int(train_cfg.get('batch_size', 32)),
        environments=environments_to_scan,
        num_rotations_per_pose=int(data_cfg.get('num_rotations_per_pose', 2)),
        frames_per_sequence=int(data_cfg.get('frames_per_sequence', 10)),
        num_workers=int(data_cfg.get('num_workers', 4)),
        center_spec=data_cfg.get('center_spec', 0),
        # Protocol & remap
        split_cfg=split_cfg,
        axis_remap_cfg=axis_cfg
    )

    # Save a copy of the effective config to the run directory
    try:
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.warning(f"Failed to save config copy: {e}")

    logger.info("Data loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    logger.info(f"Run directory: {run_dir}")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=train_cfg.get('num_epochs', 100),
        save_dir=run_dir,
        save_best=ckpt_cfg.get('save_best', True),
        save_last=ckpt_cfg.get('save_last', True),
        save_interval=cfg_mgr.get('logging.save_interval', 10)
    )

    logger.info("Training completed successfully!")
    logger.info(f"Final train loss: {history['train_total_loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history['val_total_loss'][-1]:.4f}")
    logger.info(f"Final pose error: {history['val_pose_error_mm'][-1]:.1f}mm")
    logger.info(f"Final rotation error: {history['val_rotation_error_deg'][-1]:.1f}°")

    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info("Test set results:")
    for key, value in test_metrics.items():
        if key in ['mpjpe', 'pampjpe']:
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value:.4f}")