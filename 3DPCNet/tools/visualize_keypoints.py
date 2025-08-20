"""
Visualize 3D keypoints with their index IDs to verify joint ordering.

Usage:
  python visualize_keypoints.py --input test_data/test_samples.npy --out plots/keypoints_indices.png

Notes:
  - Expects shape (N, J, 3) or (J, 3). If (N, J, 3), uses the first sample.
  - Applies the same axis remap used in inference plots: [x,y,z] -> [z, -x, -y]
  - Draws point index labels next to each joint.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def visualize_keypoints(points: np.ndarray, save_path: str):
    if points.ndim == 3:
        points = points[0]
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be (J, 3) or (N, J, 3)"

    # Remap axes for plotting: [x, y, z] -> [z, -x, -y]
    pts = points[:, [2, 0, 1]].copy()
    pts[:, 1] = -pts[:, 1]
    pts[:, 2] = -pts[:, 2]

    # High-res figure (approx 2K)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', s=20, depthshade=False)
    for idx, (x, y, z) in enumerate(pts):
        ax.text(x, y, z, str(idx), color='k', fontsize=8)

    ax.set_title('3D Keypoints with Indices')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    _set_axes_equal(ax)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D keypoints with indices')
    parser.add_argument('--input', type=str, required=True, help='Path to .npy or .npz (expects (N,J,3) or (J,3))')
    parser.add_argument('--out', type=str, default='plots/keypoints_indices.png', help='Output image path')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f'Input not found: {args.input}')

    if args.input.endswith('.npz'):
        data = np.load(args.input)
        # Try common keys
        if 'canonical' in data:
            arr = data['canonical']
        elif 'input' in data:
            arr = data['input']
        else:
            # Fallback: first array in the npz
            first_key = list(data.keys())[0]
            arr = data[first_key]
    else:
        arr = np.load(args.input)

    visualize_keypoints(arr, args.out)


if __name__ == '__main__':
    main()


