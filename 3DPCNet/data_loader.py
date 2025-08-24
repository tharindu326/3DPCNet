"""
Data loader for MM-Fi dataset for pose canonicalization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import math
from torch.utils.data import Subset
import logging
from utils.rotation_utils import create_rotation_matrix_from_euler, apply_rotation_to_pose
from utils.pose_utils import center_pose_at


class NpzPoseDataset(Dataset):
    """
    Top-level dataset for fully materialized splits (train/val/test) loaded from .npz.
    Must be defined at module scope to be picklable by DataLoader workers on Windows.
    """
    def __init__(self, d):
        self.inputs = torch.tensor(d['input_pose'], dtype=torch.float32)
        self.canon = torch.tensor(d['canonical_pose'], dtype=torch.float32)
        self.rot = torch.tensor(d['rotation_matrix'], dtype=torch.float32)
        # Optional sequence metadata
        self.env = d['env'] if 'env' in d else None
        self.subject = d['subject'] if 'subject' in d else None
        self.activity = d['activity'] if 'activity' in d else None
        self.frame_idx = d['frame_idx'] if 'frame_idx' in d else None
        self.sequence_idx = d['sequence_idx'] if 'sequence_idx' in d else None
        self.rotation_idx = d['rotation_idx'] if 'rotation_idx' in d else None

    def __len__(self):
        return int(self.inputs.shape[0])

    def __getitem__(self, i):
        sample = {
            'input_pose': self.inputs[i],
            'canonical_pose': self.canon[i],
            'rotation_matrix': self.rot[i]
        }
        if self.env is not None:
            sample['sequence_info'] = {
                'environment': str(self.env[i]),
                'subject': str(self.subject[i]) if self.subject is not None else '',
                'activity': str(self.activity[i]) if self.activity is not None else '',
                'frame_idx': int(self.frame_idx[i]) if self.frame_idx is not None else -1,
                'sequence_idx': int(self.sequence_idx[i]) if self.sequence_idx is not None else -1,
                'rotation_idx': int(self.rotation_idx[i]) if self.rotation_idx is not None else -1
            }
        return sample


@dataclass
class AxisRemapConfig:
    enabled: bool = True
    order: Tuple[int, int, int] = (2, 0, 1)  # newX=oldZ, newY=oldX, newZ=oldY
    flip: Tuple[int, int, int] = (1, -1, -1)  # multiply after reordering


@dataclass
class SplitConfig:
    setting: Optional[str] = None  # 'S1' | 'S2' | 'S3' | None
    seed: int = 42
    s1_train_ratio: float = 0.75
    val_ratio_from_train: float = 0.1
    s2_train_subjects: Optional[List[str]] = None
    s2_test_subjects: Optional[List[str]] = None
    s3_train_envs: Optional[List[str]] = None
    s3_test_envs: Optional[List[str]] = None
    load_from_path: bool = False
    # Full precomputed dataset directories (single mechanism)
    load_dir: Optional[str] = None  # directory containing train.npz/val.npz/test.npz
    save_dir: Optional[str] = None  # directory to write train.npz/val.npz/test.npz


class MMFiCanonPose(Dataset):
    """
    MM-Fi dataset loader for pose canonicalization training

    Dataset structure:
    MM-Fi/
    ├── E1/
    │   ├── S1/
    │   │   ├── A1/
    │   │   │   ├── ground_truth.npy  # Shape: (num_frames, 17, 3)
    │   │   │   └── ...
    │   │   ├── A2/
    │   │   └── ...
    │   ├── S2/
    │   └── ...
    ├── E2/
    └── ...
    """
    
    def __init__(self, 
                 data_root: str,
                 environments: Optional[List[str]] = None,
                 subjects: Optional[List[str]] = None,
                 activities: Optional[List[str]] = None,
                 frame_sampling: str = 'random',  # 'random', 'uniform', 'all'
                 frames_per_sequence: int = 1,
                 min_sequence_length: int = 30,
                 num_rotations_per_pose: int = 1,  # Number of rotated versions per canonical pose
                 center_spec: Optional[object] = None,
                 transform=None,
                 axis_remap_enabled: bool = True,
                 axis_remap_order: Tuple[int, int, int] = (2, 0, 1),
                 axis_remap_flip: Tuple[int, int, int] = (1, -1, -1)):
        """
        Initialize MM-Fi dataset for canonicalization
        Args:
            data_root: Root directory of MM-Fi dataset
            environments: List of environments to use (e.g., ['E1', 'E2'])
            subjects: List of subjects to use (e.g., ['S1', 'S2'])
            activities: List of activities to use (e.g., ['A1', 'A2'])
            frame_sampling: How to sample frames ('random', 'uniform', 'all')
            frames_per_sequence: Number of frames to sample per sequence
            min_sequence_length: Minimum sequence length to include
            num_rotations_per_pose: Number of rotated versions per canonical pose
            center_spec: Optional joint index or joint pair [i, j] to center poses at
            transform: Optional transforms to apply
            axis_remap_enabled: Whether to remap axes on load
            axis_remap_order: Reorder mapping for axes (new <- old indices)
            axis_remap_flip: Per-axis sign flips after reorder
        """
        self.data_root = Path(data_root)
        self.frame_sampling = frame_sampling
        self.frames_per_sequence = frames_per_sequence
        self.min_sequence_length = min_sequence_length
        self.num_rotations_per_pose = num_rotations_per_pose
        self.transform = transform
        # Centering: int index or pair [i, j] (e.g., [11, 12] for pelvis midpoint)
        self.center_spec = center_spec if center_spec is not None else [11, 12]
        # Axis remap config
        self.axis_remap_enabled = bool(axis_remap_enabled)
        self.axis_remap_order = axis_remap_order
        self.axis_remap_flip = axis_remap_flip
        
        self.logger = logging.getLogger(__name__)
        
        # Determine discovery/default values
        if environments is None:
            # Discover envs under data_root
            try:
                environments = [d.name for d in sorted(self.data_root.iterdir()) if d.is_dir() and d.name.upper().startswith('E')]
            except Exception:
                environments = []
        # Keep subjects/activities as passed; if None, they will be discovered per env in _load_sequences
        self.environments = environments
        self.subjects = subjects
        self.activities = activities
        
        self.logger.info(
            f"MMFiCanonPose init: center_spec={self.center_spec}, "
            f"axis_remap_enabled={getattr(self, 'axis_remap_enabled', True)}, "
            f"axis_order={getattr(self, 'axis_remap_order', (2,0,1))}, axis_flip={getattr(self, 'axis_remap_flip', (1,-1,-1))}, "
            f"frames_per_sequence={self.frames_per_sequence}, num_rotations_per_pose={self.num_rotations_per_pose}, "
            f"frame_sampling={self.frame_sampling}, data_root={str(self.data_root)}, environments={self.environments}"
        )

        self.sequences = self._load_sequences()
        
        # Create frame indices for sampling (considering rotations)
        self.frame_indices = self._create_frame_indices()
        
        self.logger.info(f"Loaded {len(self.sequences)} sequences with {len(self.frame_indices)} total training samples")
    
    def generate_training_sample(self, canonical_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate training sample by applying random rotation to canonical pose
        
        Angles and order:
        - yaw: rotation around Z (turn left/right)
        - pitch: rotation around Y (look up/down)
        - roll: rotation around X (tilt/horizon)
        Your create_rotation_matrix_from_euler expects [pitch, yaw, roll], and builds
        R = Rz(yaw) · Ry(pitch) · Rx(roll).
        
        Mixture used in this function (one draw per sample):
        - Base (75%):
          yaw ∼ Uniform(−70°, +70°) → front-ish views
          pitch ∼ N(0, 12°) clipped to [−20°, +20°] → mild up/down
          roll ∼ N(0, 4°) clipped to [−8°, +8°] → small horizon tilt
        - Profile (15%):
          yaw ≈ ±90° ± 10° → strong side views
          pitch ∼ N(0, 8°) clip [−15°, +15°], roll ∼ N(0, 3°) clip [−6°, +6°]
        - Back (10%):
          yaw ≈ 180° ± 20° → back views; pitch/roll small as above
        
        Hard bounds (after sampling):
        - |pitch| ≤ 25° to avoid ground/ceiling angles
        - |roll|  ≤ 12° to avoid unrealistic horizon tilt
        This prevents "camera-on-the-floor" or upside‑down views while keeping useful variation.
        Args:
            canonical_pose: (17, 3) canonical 3D pose
        Returns:
            rotated_pose: Input pose with random rotation (17, 3)
            canonical_pose: Target canonical pose (17, 3) - unchanged
            rotation_matrix: Applied rotation matrix (3, 3)
        """
        # Add batch dimension for processing
        canonical_pose_batch = canonical_pose.unsqueeze(0)  # (1, 17, 3)
        # Center canonical pose around specified joint or joint pair
        canonical_pose_centered, center_point = center_pose_at(canonical_pose_batch, self.center_spec)
        
        # Generate practical camera-like angles (yaw around Z, pitch around Y, roll around X)
        # Mixture: 75% base, 15% profile (±90°), 10% back (~180°)
        mode = torch.rand(1).item()
        if mode < 0.75:
            yaw_deg = float(torch.empty(1).uniform_(-70.0, 70.0))
            pitch_deg = float(torch.clamp(torch.randn(1) * 12.0, -20.0, 20.0))
            roll_deg = float(torch.clamp(torch.randn(1) * 4.0,  -8.0,  8.0))
        elif mode < 0.90:
            yaw_deg = (90.0 if torch.rand(1).item() < 0.5 else -90.0) + float(torch.randn(1) * 10.0)
            pitch_deg = float(torch.clamp(torch.randn(1) * 8.0,  -15.0, 15.0))
            roll_deg = float(torch.clamp(torch.randn(1) * 3.0,   -6.0,  6.0))
        else:
            yaw_deg = 180.0 + float(torch.randn(1) * 20.0)
            pitch_deg = float(torch.clamp(torch.randn(1) * 8.0,  -15.0, 15.0))
            roll_deg = float(torch.clamp(torch.randn(1) * 3.0,   -6.0,  6.0))

        # Hard bounds (reject extreme up/down or tilted cameras)
        pitch_deg = max(-25.0, min(25.0, pitch_deg))
        roll_deg  = max(-12.0, min(12.0,  roll_deg))

        # Convert to radians; create_rotation_matrix_from_euler expects [roll (X), pitch (Y), yaw (Z)]
        angles = torch.tensor([
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg)
        ], dtype=torch.float32)
        
        # Create rotation matrix
        rotation_matrix = create_rotation_matrix_from_euler(angles)
        rotation_matrix_batch = rotation_matrix.unsqueeze(0)  # (1, 3, 3)
        
        # Apply rotation to centered canonical pose
        rotated_pose_batch = apply_rotation_to_pose(canonical_pose_centered, rotation_matrix_batch)
        rotated_pose = rotated_pose_batch.squeeze(0)  # (17, 3)
        
        # Return centered canonical pose as ground truth
        
        return rotated_pose, canonical_pose_centered.squeeze(0), rotation_matrix
    
    def _load_sequences(self) -> List[Dict]:
        """Load all valid sequences from the dataset"""
        sequences = []
        for env in self.environments:
            env_path = self.data_root / env
            if not env_path.exists():
                self.logger.warning(f"[LoadSequences] Environment path missing: {env_path}")
                continue
            # Discover subjects if not provided
            if self.subjects is None:
                subj_list = [d.name for d in sorted(env_path.iterdir()) if d.is_dir() and d.name.upper().startswith('S')]
            else:
                subj_list = list(self.subjects)
            if not subj_list:
                self.logger.warning(f"[LoadSequences] No subjects found under {env_path}")
            for subj in subj_list:
                subj_path = env_path / subj
                if not subj_path.exists():
                    continue
                # Discover activities if not provided
                if self.activities is None:
                    act_list = [d.name for d in sorted(subj_path.iterdir()) if d.is_dir() and d.name.upper().startswith('A')]
                else:
                    act_list = list(self.activities)
                if not act_list:
                    self.logger.warning(f"[LoadSequences] No activities found under {subj_path}")
                for act in act_list:
                    act_path = subj_path / act
                    gt_file = act_path / 'ground_truth.npy'
                    
                    if gt_file.exists():
                        try:
                            # Load and validate the sequence
                            keypoints_data = np.load(gt_file)
                            
                            if len(keypoints_data.shape) == 3 and keypoints_data.shape[1:] == (17, 3):
                                num_frames = keypoints_data.shape[0]
                                
                                if num_frames >= self.min_sequence_length:
                                    sequences.append({
                                        'file_path': str(gt_file),
                                        'environment': env,
                                        'subject': subj,
                                        'activity': act,
                                        'num_frames': num_frames,
                                        'sequence_idx': len(sequences)
                                    })
                                else:
                                    self.logger.warning(f"Sequence {env}/{subj}/{act} too short: {num_frames} frames")
                            else:
                                self.logger.warning(f"Invalid shape in {env}/{subj}/{act}: {keypoints_data.shape}")
                                
                        except Exception as e:
                            self.logger.error(f"Error loading {env}/{subj}/{act}: {e}")
        
        return sequences
    
    def _create_frame_indices(self) -> List[Tuple[int, int, int]]:
        """Create list of (sequence_idx, frame_idx, rotation_idx) tuples for sampling"""
        frame_indices = []
        
        for seq_idx, seq_info in enumerate(self.sequences):
            num_frames = seq_info['num_frames']
            
            if self.frame_sampling == 'all':
                # Use all frames
                for frame_idx in range(num_frames):
                    for rot_idx in range(self.num_rotations_per_pose):
                        frame_indices.append((seq_idx, frame_idx, rot_idx))
                        
            elif self.frame_sampling == 'uniform':
                # Sample uniformly across the sequence
                if self.frames_per_sequence >= num_frames:
                    frame_idxs = list(range(num_frames))
                else:
                    frame_idxs = np.linspace(0, num_frames-1, self.frames_per_sequence, dtype=int)
                
                for frame_idx in frame_idxs:
                    for rot_idx in range(self.num_rotations_per_pose):
                        frame_indices.append((seq_idx, frame_idx, rot_idx))
                        
            elif self.frame_sampling == 'random':
                # Will sample randomly during __getitem__
                for _ in range(self.frames_per_sequence):
                    for rot_idx in range(self.num_rotations_per_pose):
                        frame_indices.append((seq_idx, -1, rot_idx))  # -1 indicates random sampling
        
        return frame_indices
    
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.frame_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample
        Returns:
            Dictionary containing:
                - 'input_pose': (17, 3) rotated 3D pose (network input)
                - 'canonical_pose': (17, 3) canonical 3D pose (ground truth)
                - 'rotation_matrix': (3, 3) applied rotation matrix (ground truth)
                - 'sequence_info': metadata about the sequence
        """
        seq_idx, frame_idx, rot_idx = self.frame_indices[idx]
        seq_info = self.sequences[seq_idx]
        
        keypoints_data = np.load(seq_info['file_path'])
        
        # Sample frame
        if frame_idx == -1:  # Random sampling
            frame_idx = np.random.randint(0, seq_info['num_frames'])
        
        # Get the canonical pose for this frame
        keypoints_orig = keypoints_data[frame_idx]  # Shape: (17, 3)
        # Optional axis remap for original dataset format to canonical axes
        if getattr(self, 'axis_remap_enabled', True):
            kp = keypoints_orig.copy()
            order = self.axis_remap_order if hasattr(self, 'axis_remap_order') else (2, 0, 1)
            flip = self.axis_remap_flip if hasattr(self, 'axis_remap_flip') else (1, -1, -1)
            remapped = kp[:, [order[0], order[1], order[2]]]
            remapped = remapped * np.array(flip, dtype=remapped.dtype)
            keypoints_orig = remapped
        canonical_pose = torch.tensor(keypoints_orig, dtype=torch.float32)
        
        # Apply transforms to canonical pose if provided
        if self.transform:
            canonical_pose = self.transform(canonical_pose)
        
        # Generate training sample (rotated input + canonical GT)
        input_pose, canonical_pose_gt, rotation_matrix = self.generate_training_sample(canonical_pose)
        
        return {
            'input_pose': input_pose,           # Network input: rotated pose
            'canonical_pose': canonical_pose_gt, # Ground truth: canonical pose
            'rotation_matrix': rotation_matrix,  # Ground truth: rotation matrix
            'sequence_info': {
                'environment': seq_info['environment'],
                'subject': seq_info['subject'],
                'activity': seq_info['activity'],
                'frame_idx': frame_idx,
                'sequence_idx': seq_idx,
                'rotation_idx': rot_idx
            }
        }
    
    def get_sequence_info(self) -> List[Dict]:
        """Get information about all loaded sequences"""
        return self.sequences.copy()
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        total_frames = sum(seq['num_frames'] for seq in self.sequences)
        
        env_counts = {}
        subj_counts = {}
        act_counts = {}
        
        for seq in self.sequences:
            env_counts[seq['environment']] = env_counts.get(seq['environment'], 0) + 1
            subj_counts[seq['subject']] = subj_counts.get(seq['subject'], 0) + 1
            act_counts[seq['activity']] = act_counts.get(seq['activity'], 0) + 1
        
        return {
            'total_sequences': len(self.sequences),
            'total_frames': total_frames,
            'total_samples': len(self.frame_indices),
            'environments': env_counts,
            'subjects': subj_counts,
            'activities': act_counts,
            'avg_frames_per_sequence': total_frames / len(self.sequences) if self.sequences else 0
        }


def create_train_val_split():
    """
    Deprecated: S1/S2/S3 protocols are used exclusively. This placeholder remains to avoid import errors.
    """
    raise NotImplementedError("Random percentage splits are removed. Use S1/S2/S3 via split_cfg.")


def create_data_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 42,
    split_cfg: Optional[SplitConfig] = None,
    axis_remap_cfg: Optional[AxisRemapConfig] = None,
    environments: Optional[List[str]] = None,
    activities: Optional[List[str]] = None,
    frame_sampling: str = 'random',
    frames_per_sequence: int = 1,
    min_sequence_length: int = 30,
    num_rotations_per_pose: int = 1,
    center_spec: Optional[object] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test data loaders using S1/S2/S3 split protocols
    
    Args:
        data_root: Root directory of MM-Fi dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        random_seed: Random seed for reproducible splits
        **dataset_kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Option 0: Full materialized dataset loading (single mechanism)
    if split_cfg and split_cfg.load_from_path:
        if not split_cfg.load_dir:
            raise FileNotFoundError("[FullSplit] load_from_path is true but load_dir is empty")
        # Load fully materialized datasets from load_dir
        dir_path = split_cfg.load_dir
        train_npz = os.path.join(dir_path, 'train.npz')
        val_npz = os.path.join(dir_path, 'val.npz')
        test_npz = os.path.join(dir_path, 'test.npz')
        if not (os.path.exists(train_npz) and os.path.exists(val_npz) and os.path.exists(test_npz)):
            raise FileNotFoundError(f"[FullSplit] Missing train/val/test npz in {dir_path}")
        logging.getLogger(__name__).info(f"[FullSplit] Loading precomputed datasets from {dir_path}")
        def _loader_from_npz(path):
            data = np.load(path, allow_pickle=False)
            return NpzPoseDataset(data)
        train_ds = _loader_from_npz(train_npz)
        val_ds = _loader_from_npz(val_npz)
        test_ds = _loader_from_npz(test_npz)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader

    else:
        # Always use S1/S2/S3 protocols; default to S1 if not specified
        setting = (split_cfg.setting if split_cfg and split_cfg.setting in {'S1', 'S2', 'S3'} else 'S1')
        seed = split_cfg.seed if split_cfg else random_seed
        logging.getLogger(__name__).info(f"[Splits] Generating splits with setting={setting}, seed={seed}")

        # Build a single full dataset of sequences
        full_dataset = MMFiCanonPose(
            data_root=data_root,
            environments=environments,
            subjects=None,
            activities=activities,
            frame_sampling=frame_sampling,
            frames_per_sequence=frames_per_sequence,
            min_sequence_length=min_sequence_length,
            num_rotations_per_pose=num_rotations_per_pose,
            center_spec=center_spec,
            transform=None,
            axis_remap_enabled=(axis_remap_cfg.enabled if axis_remap_cfg else True),
            axis_remap_order=(axis_remap_cfg.order if axis_remap_cfg else (2,0,1)),
            axis_remap_flip=(axis_remap_cfg.flip if axis_remap_cfg else (1,-1,-1)),
        )
        sequences = full_dataset.get_sequence_info()
        rng = np.random.RandomState(seed)

        # Helper: map from set of seq indices to sample indices
        def sample_indices_for_sequences(seq_idx_set):
            idxs = []
            for i, (seq_idx, frame_idx, rot_idx) in enumerate(full_dataset.frame_indices):
                if seq_idx in seq_idx_set:
                    idxs.append(i)
            return idxs

        if setting == 'S1':
            # Random split over sequences: train:test = s1_ratio : (1-s1_ratio)
            all_seq_idx = np.arange(len(sequences))
            rng.shuffle(all_seq_idx)
            s1_ratio = split_cfg.s1_train_ratio if split_cfg and split_cfg.s1_train_ratio is not None else 0.75
            val_ratio_from_train = split_cfg.val_ratio_from_train if split_cfg and split_cfg.val_ratio_from_train is not None else 0.1
            n_train_total = int(np.floor(s1_ratio * len(sequences)))
            train_seq = all_seq_idx[:n_train_total]
            test_seq = all_seq_idx[n_train_total:]
            # carve validation from train sequences
            n_val = int(np.floor(val_ratio_from_train * len(train_seq)))
            val_seq = train_seq[:n_val]
            train_seq = train_seq[n_val:]
            logging.getLogger(__name__).info(f"[Splits:S1] ratio={s1_ratio}, val_from_train={val_ratio_from_train}; train_seq={len(train_seq)}, val_seq={len(val_seq)}, test_seq={len(test_seq)}")

        elif setting == 'S2':
            # Cross-subject: select 32 train, 8 test (deterministic by seed) unless provided explicitly
            subjects = sorted({s['subject'] for s in sequences})
            if split_cfg and (split_cfg.s2_train_subjects or split_cfg.s2_test_subjects):
                train_subjects = set(split_cfg.s2_train_subjects or [])
                test_subjects = set(split_cfg.s2_test_subjects or [])
                if not train_subjects:
                    rng.shuffle(subjects)
                    train_subjects = set(subjects[:32])
                if not test_subjects:
                    remaining = [s for s in subjects if s not in train_subjects]
                    test_subjects = set(remaining[:8])
            else:
                rng.shuffle(subjects)
                train_subjects = set(subjects[:32])
                test_subjects = set(subjects[32:40])
            train_seq = [s['sequence_idx'] for s in sequences if s['subject'] in train_subjects]
            test_seq = [s['sequence_idx'] for s in sequences if s['subject'] in test_subjects]
            rng.shuffle(train_seq)
            val_ratio_from_train = split_cfg.val_ratio_from_train if split_cfg and split_cfg.val_ratio_from_train is not None else 0.1
            n_val = int(np.floor(val_ratio_from_train * len(train_seq)))
            val_seq = train_seq[:n_val]
            train_seq = train_seq[n_val:]
            logging.getLogger(__name__).info(f"[Splits:S2] subjects_train={len(train_subjects)}, subjects_test={len(test_subjects)}; train_seq={len(train_seq)}, val_seq={len(val_seq)}, test_seq={len(test_seq)}")

        else:  # S3
            # Cross-environment: randomly select 3 environments for train and 1 for test unless provided explicitly
            envs = [s['environment'] for s in sequences]
            unique_envs = sorted(set(envs))
            if split_cfg and (split_cfg.s3_train_envs or split_cfg.s3_test_envs):
                train_envs = set(split_cfg.s3_train_envs or [])
                test_envs = set(split_cfg.s3_test_envs or [])
                if not train_envs or not test_envs:
                    # fill missing parts from available envs deterministically
                    rng.shuffle(unique_envs)
                    if not train_envs:
                        train_envs = set(unique_envs[:min(3, len(unique_envs))])
                    if not test_envs:
                        remaining = [e for e in unique_envs if e not in train_envs]
                        test_envs = set(remaining[:1])
            else:
                rng.shuffle(unique_envs)
                train_envs = set(unique_envs[:min(3, len(unique_envs))])
                remaining = [e for e in unique_envs if e not in train_envs]
                test_envs = set(remaining[:1])
            train_seq = [s['sequence_idx'] for s in sequences if s['environment'] in train_envs]
            test_seq = [s['sequence_idx'] for s in sequences if s['environment'] in test_envs]
            rng.shuffle(train_seq)
            val_ratio_from_train = split_cfg.val_ratio_from_train if split_cfg and split_cfg.val_ratio_from_train is not None else 0.1
            n_val = int(np.floor(val_ratio_from_train * len(train_seq)))
            val_seq = train_seq[:n_val]
            train_seq = train_seq[n_val:]
            logging.getLogger(__name__).info(f"[Splits:S3] train_envs={sorted(list(train_envs))}, test_envs={sorted(list(test_envs))}; total_seq={len(sequences)}, train_seq={len(train_seq)}, val_seq={len(val_seq)}, test_seq={len(test_seq)}")

        # Convert to sample indices
        train_indices = sample_indices_for_sequences(set(train_seq))
        val_indices = sample_indices_for_sequences(set(val_seq))
        test_indices = sample_indices_for_sequences(set(test_seq))

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        test_subset = Subset(full_dataset, test_indices)
        # Optional: save fully materialized datasets if requested
        if split_cfg and split_cfg.save_dir:
            try:
                os.makedirs(split_cfg.save_dir, exist_ok=True)
                def _dump(path, indices):
                    if not indices:
                        # Handle empty split case
                        logging.getLogger(__name__).warning(f"[FullSplit] Empty split for {os.path.basename(path)}, creating empty arrays")
                        np.savez(path,
                                 input_pose=np.empty((0, 17, 3), dtype=np.float32),
                                 canonical_pose=np.empty((0, 17, 3), dtype=np.float32),
                                 rotation_matrix=np.empty((0, 3, 3), dtype=np.float32),
                                 env=np.array([], dtype='<U10'),
                                 subject=np.array([], dtype='<U10'),
                                 activity=np.array([], dtype='<U10'),
                                 frame_idx=np.array([], dtype=np.int64),
                                 sequence_idx=np.array([], dtype=np.int64),
                                 rotation_idx=np.array([], dtype=np.int64))
                        return
                    
                    inputs = []
                    canons = []
                    rots = []
                    env = []
                    subject = []
                    activity = []
                    frame_idx_arr = []
                    seq_idx_arr = []
                    rot_idx_arr = []
                    for idx in indices:
                        sample = full_dataset[idx]
                        inputs.append(sample['input_pose'].numpy())
                        canons.append(sample['canonical_pose'].numpy())
                        rots.append(sample['rotation_matrix'].numpy())
                        info = sample.get('sequence_info', {})
                        env.append(info.get('environment', ''))
                        subject.append(info.get('subject', ''))
                        activity.append(info.get('activity', ''))
                        frame_idx_arr.append(info.get('frame_idx', -1))
                        seq_idx_arr.append(info.get('sequence_idx', -1))
                        rot_idx_arr.append(info.get('rotation_idx', -1))
                    np.savez(path,
                             input_pose=np.stack(inputs, axis=0),
                             canonical_pose=np.stack(canons, axis=0),
                             rotation_matrix=np.stack(rots, axis=0),
                             env=np.array(env),
                             subject=np.array(subject),
                             activity=np.array(activity),
                             frame_idx=np.array(frame_idx_arr, dtype=np.int64),
                             sequence_idx=np.array(seq_idx_arr, dtype=np.int64),
                             rotation_idx=np.array(rot_idx_arr, dtype=np.int64))
                _dump(os.path.join(split_cfg.save_dir, 'train.npz'), train_indices)
                _dump(os.path.join(split_cfg.save_dir, 'val.npz'), val_indices)
                _dump(os.path.join(split_cfg.save_dir, 'test.npz'), test_indices)
                logging.getLogger(__name__).info(f"[FullSplit] Saved train/val/test npz to {split_cfg.save_dir}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[FullSplit] Failed to save materialized datasets: {e}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader, val_loader, test_loader


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    data_root = "C:/Users/thari/Desktop/Uni-Oulu/RA-CMVS/PC-detection/E01"
    
    if not os.path.exists(data_root):
        logger.warning(f"Dataset path {data_root} not found. Update the path to test.")        
        logger.info("Usage example:")
        logger.info("""
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root="/path/to/MM-Fi",
            batch_size=32,
            num_rotations_per_pose=2
        )
        """)
        return
    
    try:
        dataset = MMFiCanonPose(
            data_root=data_root,
            environments=['E01'],  # Test with one environment
            subjects=None,  # Test with two subjects
            activities=None,  # Test with few activities
            frame_sampling='random',
            frames_per_sequence=50,
            num_rotations_per_pose=6,
        )
        
        logger.info(f"Dataset created successfully!")
        logger.info(f"Dataset size: {len(dataset)}")
        
        stats = dataset.get_statistics()
        logger.info("Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Testing data sample...")
        sample = dataset[0]
        logger.info(f"Input pose shape: {sample['input_pose'].shape}")
        logger.info(f"Canonical pose shape: {sample['canonical_pose'].shape}")
        logger.info(f"Rotation matrix shape: {sample['rotation_matrix'].shape}")
        
        logger.info("Testing DataLoader...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(dataloader))
        logger.info(f"Batch input shape: {batch['input_pose'].shape}")
        logger.info(f"Batch canonical shape: {batch['canonical_pose'].shape}")
        
        logger.info("Testing train/validation split...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_root=data_root,
            batch_size=8,
            environments=['E01'],
            subjects=None,
            activities=None,
            num_workers=0
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
                
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()