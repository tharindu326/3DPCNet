"""
Inference script for 3DPCNet pose canonicalization.
Loads a checkpoint and runs canonicalization on input poses.

python inference.py --checkpoint checkpoints/train1/best_model.pth --input test_data/test_samples.npy --save test_data/outputs.npz --plot --gt test_data/test_samples_gt.npz --plot-dir plots
"""

import argparse
import os
import torch
import yaml
import logging
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from models.model import create_pose_canonicalization_model
from utils.config_utils import ConfigManager


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    return ckpt


def run_inference(model: torch.nn.Module, poses: torch.Tensor, device: torch.device):
    model.eval()
    with torch.no_grad():
        poses = poses.to(device)
        canonical, rot_repr = model(poses)
        R = model.get_rotation_matrix(rot_repr)
        return canonical.cpu(), R.cpu()


def _get_skeleton_edges() -> List[tuple]:
    # Custom 17-joint schema:
    # 0 Pelvis; Left leg: 1-2-3; Right leg: 4-5-6; 7 Torso; 8 Neck; 9 Nose; 10 Head
    # Right arm: 11-12-13; Left arm: 14-15-16
    return [
        # Spine and head
        (0, 7), (7, 8), (8, 9), (9, 10),
        # Left leg
        (0, 1), (1, 2), (2, 3),
        # Right leg
        (0, 4), (4, 5), (5, 6),
        # Arms from neck
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16),
    ]


def _set_axes_equal(ax):
    limits = []
    for axis in [ax.get_xbound, ax.get_ybound, ax.get_zbound]:
        limits.extend(axis())
    x_min, x_max, y_min, y_max, z_min, z_max = limits
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = (x_max + x_min) / 2.0
    y_mid = (y_max + y_min) / 2.0
    z_mid = (z_max + z_min) / 2.0
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)


def _plot_skeleton(ax, pose: torch.Tensor, color: str = 'b', edges: Optional[List[tuple]] = None, label: Optional[str] = None):
    if edges is None:
        edges = _get_skeleton_edges()
    pose_np = pose.detach().cpu().numpy()
    # Data is already axis-remapped by the dataloader/create_test_samples pipeline
    ax.scatter(pose_np[:, 0], pose_np[:, 1], pose_np[:, 2], c=color, s=10, depthshade=False)
    for i, j in edges:
        xs = [pose_np[i, 0], pose_np[j, 0]]
        ys = [pose_np[i, 1], pose_np[j, 1]]
        zs = [pose_np[i, 2], pose_np[j, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=1)
    if label:
        ax.set_title(label)
    # Axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_prediction_vs_gt(pred_canonical: torch.Tensor,
                          gt_canonical: Optional[torch.Tensor],
                          input_pose: Optional[torch.Tensor],
                          save_path: str,
                          suptitle: Optional[str] = None):
    # 2K resolution (approx. 2560x1440)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=200)
    edges = _get_skeleton_edges()

    if gt_canonical is not None:
        # 1x3: Input | GT Canonical | Pred Canonical
        ax0 = fig.add_subplot(1, 3, 1, projection='3d')
        if input_pose is not None:
            _plot_skeleton(ax0, input_pose, color='b', edges=edges, label='Input Pose')
            _set_axes_equal(ax0)
        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        _plot_skeleton(ax1, gt_canonical, color='g', edges=edges, label='GT Canonical')
        _set_axes_equal(ax1)
        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        _plot_skeleton(ax2, pred_canonical, color='r', edges=edges, label='Pred Canonical')
        _set_axes_equal(ax2)
    else:
        # 1x2: Input | Pred Canonical
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        if input_pose is not None:
            _plot_skeleton(ax0, input_pose, color='b', edges=edges, label='Input Pose')
            _set_axes_equal(ax0)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        _plot_skeleton(ax, pred_canonical, color='r', edges=edges, label='Pred Canonical')
        _set_axes_equal(ax)

    if suptitle:
        fig.suptitle(suptitle)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="3DPCNet Inference")
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML (optional)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--input', type=str, default=None, help='Path to a .npy of poses with shape (N, J, 3)')
    parser.add_argument('--save', type=str, default=None, help='Path to save outputs (npz with canonical and rotation_matrix)')
    parser.add_argument('--gt', type=str, default=None, help='Optional path to GT npz (with canonical) for plotting')
    parser.add_argument('--plot', action='store_true', help='Plot predictions (and GT if provided) for all samples')
    parser.add_argument('--plot-dir', type=str, default='./plots', help='Directory to save plots')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('inference')

    cfg = ConfigManager(args.config).to_dict() if args.config else ConfigManager().to_dict()
    model_cfg = cfg.get('model', {})
    model = create_pose_canonicalization_model(
        input_joints=model_cfg.get('input_joints', 17),
        encoder_type=model_cfg.get('encoder_type', 'mlp'),
        rotation_type=model_cfg.get('rotation_type', '6d'),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        encoder_output_dim=model_cfg.get('encoder_output_dim', 256),
        dropout=model_cfg.get('dropout', 0.1),
        predict_mode=model_cfg.get('predict_mode', 'rotation_only')
    ).to(device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, device)

    if args.input is None:
        logger.info("No input provided. Running a dummy example (batch=1)")
        poses = torch.randn(1, model_cfg.get('input_joints', 17), 3)
    else:
        import numpy as np
        poses_np = np.load(args.input)
        assert poses_np.ndim == 3 and poses_np.shape[2] == 3, "Input must be (N, J, 3)"
        poses = torch.tensor(poses_np, dtype=torch.float32)

    canonical, R = run_inference(model, poses, device)
    logger.info(f"Inference done. canonical shape: {tuple(canonical.shape)}, R shape: {tuple(R.shape)}")

    if args.save:
        import numpy as np
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        np.savez(args.save, canonical=canonical.numpy(), rotation_matrix=R.numpy())
        logger.info(f"Saved outputs to {args.save}")

    # Optional plotting
    if args.plot:
        gt_canon = None
        if args.gt is not None and os.path.exists(args.gt):
            try:
                import numpy as np
                gt_npz = np.load(args.gt)
                if 'canonical' in gt_npz:
                    gt_canon = torch.tensor(gt_npz['canonical'], dtype=torch.float32)
                else:
                    logger.warning('GT file does not contain "canonical". Skipping GT overlays.')
            except Exception as e:
                logger.warning(f'Failed to load GT file {args.gt}: {e}')

        # Plot all samples
        for i in range(canonical.shape[0]):
            pred_i = canonical[i]
            gt_i = gt_canon[i] if gt_canon is not None and i < gt_canon.shape[0] else None
            inp_i = poses[i].cpu() if isinstance(poses, torch.Tensor) else None
            out_path = os.path.join(args.plot_dir, f'pred_vs_gt_{i:03d}.png')
            plot_prediction_vs_gt(pred_i, gt_i, inp_i, out_path, suptitle=f'Sample {i}')
        logger.info(f'Saved {canonical.shape[0]} plot(s) to {args.plot_dir}')


if __name__ == '__main__':
    main()


