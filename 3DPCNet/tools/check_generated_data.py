"""
Simple script to load test split data and visualize random samples.
Usage: python check_generated_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random

def load_test_data(test_dir="dataset/splits/S3_split_250824_2"):
    """Load test data from npz file"""
    test_file = os.path.join(test_dir, "test.npz")
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Available files in directory:")
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                print(f"  {f}")
        return None
    
    print(f"Loading test data from: {test_file}")
    data = np.load(test_file)
    
    print("Available keys:", list(data.keys()))
    print("Data shapes:")
    for key in data.keys():
        print(f"  {key}: {data[key].shape}")
    
    return data

def get_skeleton_edges():
    """Define skeleton connections for visualization"""
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

def plot_skeleton(ax, pose, color='b', alpha=1.0, title=""):
    """Plot 3D skeleton"""
    edges = get_skeleton_edges()
    
    # Plot joints
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
              c=color, s=10, alpha=alpha, depthshade=False)
    
    # Plot bones
    for i, j in edges:
        xs = [pose[i, 0], pose[j, 0]]
        ys = [pose[i, 1], pose[j, 1]]
        zs = [pose[i, 2], pose[j, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=alpha)
    
    # Set equal aspect ratio
    max_range = np.array([pose[:, 0].max() - pose[:, 0].min(),
                         pose[:, 1].max() - pose[:, 1].min(),
                         pose[:, 2].max() - pose[:, 2].min()]).max() / 2.0
    mid_x = (pose[:, 0].max() + pose[:, 0].min()) * 0.5
    mid_y = (pose[:, 1].max() + pose[:, 1].min()) * 0.5
    mid_z = (pose[:, 2].max() + pose[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

def visualize_samples(data, num_samples=6):
    """Visualize random samples"""
    total_samples = data['input_pose'].shape[0]
    print(f"Total test samples: {total_samples}")
    
    if total_samples == 0:
        print("No test samples found!")
        return
    
    # Select random samples
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(sample_indices):
        # Get data for this sample
        input_pose = data['input_pose'][idx]
        canonical_pose = data['canonical_pose'][idx]
        rotation_matrix = data['rotation_matrix'][idx]
        
        # Sample info
        env = data['env'][idx] if 'env' in data else 'N/A'
        subject = data['subject'][idx] if 'subject' in data else 'N/A'
        activity = data['activity'][idx] if 'activity' in data else 'N/A'
        
        print(f"Sample {idx}: {env}/{subject}/{activity}")
        print(f"  Input pose range: [{input_pose.min():.3f}, {input_pose.max():.3f}]")
        print(f"  Canonical pose range: [{canonical_pose.min():.3f}, {canonical_pose.max():.3f}]")
        print(f"  Rotation matrix det: {np.linalg.det(rotation_matrix):.3f}")
        
        # Plot input and canonical side by side
        ax1 = fig.add_subplot(2, num_samples, i + 1, projection='3d')
        plot_skeleton(ax1, input_pose, color='red', title=f"Input #{idx}")
        
        ax2 = fig.add_subplot(2, num_samples, i + 1 + num_samples, projection='3d')
        plot_skeleton(ax2, canonical_pose, color='blue', title=f"Canonical #{idx}")
    
    plt.tight_layout()
    plt.savefig('test_samples_check.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization as 'test_samples_check.png'")

def check_data_consistency(data):
    """Check data consistency"""
    print("\n=== Data Consistency Check ===")
    
    input_poses = data['input_pose']
    canonical_poses = data['canonical_pose']
    rotation_matrices = data['rotation_matrix']
    
    print(f"Input poses shape: {input_poses.shape}")
    print(f"Canonical poses shape: {canonical_poses.shape}")
    print(f"Rotation matrices shape: {rotation_matrices.shape}")
    
    # Check for NaN or inf values
    print(f"Input poses - NaN: {np.isnan(input_poses).sum()}, Inf: {np.isinf(input_poses).sum()}")
    print(f"Canonical poses - NaN: {np.isnan(canonical_poses).sum()}, Inf: {np.isinf(canonical_poses).sum()}")
    print(f"Rotation matrices - NaN: {np.isnan(rotation_matrices).sum()}, Inf: {np.isinf(rotation_matrices).sum()}")
    
    # Check rotation matrix properties
    dets = np.linalg.det(rotation_matrices)
    print(f"Rotation matrix determinants - Mean: {dets.mean():.3f}, Std: {dets.std():.3f}")
    print(f"Rotation matrix determinants - Min: {dets.min():.3f}, Max: {dets.max():.3f}")
    
    # Check if rotation matrices are orthogonal
    should_be_identity = np.matmul(rotation_matrices, np.transpose(rotation_matrices, (0, 2, 1)))
    identity = np.eye(3)[None, :, :].repeat(len(rotation_matrices), axis=0)
    ortho_error = np.mean(np.abs(should_be_identity - identity))
    print(f"Orthogonality error (should be ~0): {ortho_error:.6f}")
    
    # Environment/subject/activity distribution
    if 'env' in data:
        envs, env_counts = np.unique(data['env'], return_counts=True)
        print(f"Environments: {dict(zip(envs, env_counts))}")
    
    if 'subject' in data:
        subjects, subj_counts = np.unique(data['subject'], return_counts=True)
        print(f"Subjects: {len(subjects)} unique ({subjects[:5]}...)")
    
    if 'activity' in data:
        activities, act_counts = np.unique(data['activity'], return_counts=True)
        print(f"Activities: {len(activities)} unique ({activities[:5]}...)")

def main():
    print("=== Test Data Checker ===")
    
    # Load test data
    data = load_test_data()
    if data is None:
        return
    
    # Check data consistency
    check_data_consistency(data)
    
    # Visualize random samples
    print("\n=== Visualizing Random Samples ===")
    visualize_samples(data, num_samples=6)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
