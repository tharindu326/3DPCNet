"""
Utility to create random test samples for inference using the data loader.
Saves an (N, J, 3) numpy array of input poses (rotated) for use with inference.py.
Optionally saves ground-truth canonical poses and rotation matrices.

Usage:
python create_test_samples.py --config checkpoints/train1/base_config.yaml --split test --num-samples 20 --out-dir test_data --save-gt

"""

import argparse
import os
import numpy as np
import torch
import logging

from utils.config_utils import ConfigManager
from data_loader import create_data_loaders, AxisRemapConfig, SplitConfig


def collect_samples(loader, num_samples: int):
    inputs = []
    canonicals = []
    rotations = []

    for batch in loader:
        inputs.append(batch['input_pose'].cpu().numpy())
        canonicals.append(batch['canonical_pose'].cpu().numpy())
        rotations.append(batch['rotation_matrix'].cpu().numpy())
        if sum(arr.shape[0] for arr in inputs) >= num_samples:
            break

    if not inputs:
        raise RuntimeError("No samples collected. Check dataset path and filters.")

    inputs = np.concatenate(inputs, axis=0)[:num_samples]
    canonicals = np.concatenate(canonicals, axis=0)[:num_samples]
    rotations = np.concatenate(rotations, axis=0)[:num_samples]
    return inputs, canonicals, rotations


def main():
    parser = argparse.ArgumentParser(description="Create random test samples for inference")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config (uses base if omitted)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Which split to draw from')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of samples to generate')
    parser.add_argument('--out-dir', type=str, default='test_data', help='Output directory to save files into')
    parser.add_argument('--save-gt', action='store_true', help='Also save canonical poses and rotation matrices to a .npz')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('create_test_samples')

    cfg = ConfigManager(args.config).to_dict() if args.config else ConfigManager().to_dict()
    train_cfg = cfg.get('training', {})
    data_cfg = cfg.get('data', {})
    # Build configs
    axis_cfg_dict = data_cfg.get('axis_remap', {})
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

    if not data_cfg.get('train_data_path'):
        raise FileNotFoundError('data.train_data_path must be set in the config')

    logger.info('Building data loaders...')
    # Determine environments to scan consistently with training script
    environments_to_scan = None
    if split_cfg.setting == 'S3':
        train_envs = split_cfg.s3_train_envs or []
        test_envs = split_cfg.s3_test_envs or []
        union_envs = sorted(list({*train_envs, *test_envs}))
        environments_to_scan = union_envs if len(union_envs) > 0 else None
    elif split_cfg.setting in {'S1', 'S2'}:
        environments_to_scan = None

    train_loader, val_loader, test_loader = create_data_loaders(
        data_root=data_cfg.get('train_data_path'),
        batch_size=int(train_cfg.get('batch_size', 32)),
        environments=environments_to_scan,
        num_rotations_per_pose=int(data_cfg.get('num_rotations_per_pose', 2)),
        frames_per_sequence=int(data_cfg.get('frames_per_sequence', 10)),
        num_workers=int(data_cfg.get('num_workers', 4)),
        center_spec=data_cfg.get('center_spec', 0),
        split_cfg=split_cfg,
        axis_remap_cfg=axis_cfg
    )

    loader_map = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    loader = loader_map[args.split]

    logger.info(f'Collecting {args.num_samples} samples from {args.split} split...')
    inputs, canonicals, rotations = collect_samples(loader, args.num_samples)

    # Save to directory
    os.makedirs(args.out_dir, exist_ok=True)
    inputs_path = os.path.join(args.out_dir, 'test_samples.npy')
    np.save(inputs_path, inputs)
    logger.info(f'Saved inputs to {inputs_path} with shape {inputs.shape}')

    if args.save_gt:
        gt_path = os.path.join(args.out_dir, 'test_samples_gt.npz')
        np.savez(gt_path, canonical=canonicals, rotation_matrix=rotations)
        logger.info(f'Saved ground-truth to {gt_path}')


if __name__ == '__main__':
    main()


