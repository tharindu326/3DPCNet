"""
PCNet: Pose Canonicalization Network
A PyTorch implementation for 3D pose canonicalization using various encoder architectures.
"""

from .models.model import PoseCanonicalizationNet, create_pose_canonicalization_model
from .utils.config_utils import load_config, merge_configs
from .train import PCTrainer

__version__ = "1.0.0"
__author__ = "tharindu"

__all__ = [
    "PoseCanonicalizationNet",
    "create_pose_canonicalization_model", 
    "PCTrainer",
    "load_config",
    "merge_configs"
]