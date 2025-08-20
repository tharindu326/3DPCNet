import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicLoss(nn.Module):
    """
    Geodesic loss for rotation matrices (adapted from 6DRepNet)
    """
    def __init__(self, eps=1e-7, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, m1, m2):
        """
        Compute geodesic distance between rotation matrices
        Args:
            m1, m2: (batch, 3, 3) rotation matrices
        Returns:
            loss: scalar or (batch,) geodesic distances
        """
        # Relative rotation matrix
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
        # Trace calculation (adapted from 6DRepNet)
        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        # Compute geodesic distance with numerical stability
        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))
        if self.reduction == 'mean':
            return torch.mean(theta)
        elif self.reduction == 'sum':
            return torch.sum(theta)
        else:
            return theta


class PoseCanonicalizationLoss(nn.Module):
    """
    Combined loss for pose canonicalization with optional cycle and residual penalties
    """
    def __init__(self, pose_weight=1.0, rotation_weight=1.0, eps=1e-7,
                 cycle_weight: float = 0.0,
                 perceptual_weight: float = 0.0,
                 orthogonality_weight: float = 0.0,
                 residual_l2_weight: float = 0.0):
        super().__init__()
        self.pose_weight = pose_weight
        self.rotation_weight = rotation_weight
        self.pose_criterion = nn.MSELoss()
        self.rotation_criterion = GeodesicLoss(eps=eps)
        self.cycle_weight = cycle_weight
        self.perceptual_weight = perceptual_weight
        self.orthogonality_weight = orthogonality_weight
        self.residual_l2_weight = residual_l2_weight
        self.perceptual = PerceptualLoss() if perceptual_weight > 0 else None

    def _orthogonality_penalty(self, R: torch.Tensor) -> torch.Tensor:
        I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0)
        return torch.mean(torch.norm(torch.bmm(R.transpose(1, 2), R) - I, dim=(1, 2)))

    def forward(self, pred_canonical, pred_rotation, gt_canonical, gt_rotation,
                input_pose: torch.Tensor = None,
                residual: torch.Tensor = None):
        """
        Args:
            pred_canonical: (batch, J, 3)
            pred_rotation: (batch, 3, 3)
            gt_canonical: (batch, J, 3)
            gt_rotation: (batch, 3, 3)
            input_pose: (batch, J, 3) for cycle consistency (optional)
            residual: (batch, J, 3) residual term for L2 penalty (optional)
        """
        pose_loss = self.pose_criterion(pred_canonical, gt_canonical)
        rotation_loss = self.rotation_criterion(pred_rotation, gt_rotation)
        total_loss = self.pose_weight * pose_loss + self.rotation_weight * rotation_loss

        # Cycle consistency
        if self.cycle_weight > 0 and input_pose is not None:
            # Reconstruct input: R * canonical
            batch = pred_canonical.shape[0]
            canonical_flat = pred_canonical.reshape(batch, -1, 3)
            recon = torch.bmm(canonical_flat, pred_rotation.transpose(1, 2)).view_as(pred_canonical)
            cycle_loss = F.mse_loss(recon, input_pose)
            total_loss = total_loss + self.cycle_weight * cycle_loss
        else:
            cycle_loss = torch.tensor(0.0, device=pred_canonical.device)

        # Perceptual loss
        if self.perceptual is not None and self.perceptual_weight > 0:
            perc = self.perceptual(pred_canonical, gt_canonical)['total_loss']
            total_loss = total_loss + self.perceptual_weight * perc
        else:
            perc = torch.tensor(0.0, device=pred_canonical.device)

        # Orthogonality penalty
        if self.orthogonality_weight > 0:
            ortho = self._orthogonality_penalty(pred_rotation)
            total_loss = total_loss + self.orthogonality_weight * ortho
        else:
            ortho = torch.tensor(0.0, device=pred_canonical.device)

        # Residual L2 penalty
        if self.residual_l2_weight > 0 and residual is not None:
            res_l2 = torch.mean(torch.norm(residual, dim=-1))
            total_loss = total_loss + self.residual_l2_weight * res_l2
        else:
            res_l2 = torch.tensor(0.0, device=pred_canonical.device)

        return {
            'total_loss': total_loss,
            'pose_loss': pose_loss,
            'rotation_loss': rotation_loss,
            'cycle_loss': cycle_loss,
            'perceptual_loss': perc,
            'orthogonality_loss': ortho,
            'residual_l2': res_l2
        }


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for training discriminator/generator
    """
    def __init__(self, loss_type='lsgan'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown adversarial loss type: {loss_type}")
    
    def forward(self, pred, target_is_real):
        """
        Compute adversarial loss
        Args:
            pred: Discriminator predictions
            target_is_real: Whether target should be real (True) or fake (False)
        Returns:
            adversarial loss
        """
        if self.loss_type == 'lsgan':
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        else:  # vanilla
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using bone lengths and joint angles
    """
    def __init__(self, bone_weight=1.0, angle_weight=1.0):
        super().__init__()
        self.bone_weight = bone_weight
        self.angle_weight = angle_weight
        # Custom 17-joint bone connections
        self.bone_connections = [
            # Spine and head
            (0, 7), (7, 8), (8, 9), (9, 10),
            # Left leg
            (0, 1), (1, 2), (2, 3),
            # Right leg
            (0, 4), (4, 5), (5, 6),
            # Right arm from neck
            (8, 11), (11, 12), (12, 13),
            # Left arm from neck
            (8, 14), (14, 15), (15, 16)
        ]
    def compute_bone_lengths(self, pose_3d):
        """Compute bone lengths for given poses"""
        bone_lengths = []
        for joint1_idx, joint2_idx in self.bone_connections:
            joint1 = pose_3d[:, joint1_idx, :]
            joint2 = pose_3d[:, joint2_idx, :]
            length = torch.norm(joint2 - joint1, dim=-1)
            bone_lengths.append(length)
        return torch.stack(bone_lengths, dim=1)
    
    def compute_joint_angles(self, pose_3d):
        """Compute joint angles for given poses"""
        angles = []
        # Left elbow angle
        shoulder = pose_3d[:, 5, :]  # left shoulder
        elbow = pose_3d[:, 7, :]     # left elbow  
        wrist = pose_3d[:, 9, :]     # left wrist
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_angle = F.cosine_similarity(v1, v2, dim=-1)
        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        angles.append(angle)
        return torch.stack(angles, dim=1)
    
    def forward(self, pred_pose, gt_pose):
        """
        Compute perceptual loss
        Args:
            pred_pose: (batch, joints, 3) predicted poses
            gt_pose: (batch, joints, 3) ground truth poses
        Returns:
            perceptual loss
        """
        # Bone length loss
        pred_bones = self.compute_bone_lengths(pred_pose)
        gt_bones = self.compute_bone_lengths(gt_pose)
        bone_loss = F.mse_loss(pred_bones, gt_bones)
        
        # Joint angle loss
        pred_angles = self.compute_joint_angles(pred_pose)
        gt_angles = self.compute_joint_angles(gt_pose)
        angle_loss = F.mse_loss(pred_angles, gt_angles)
        
        total_loss = (self.bone_weight * bone_loss + 
                     self.angle_weight * angle_loss)
        return {
            'total_loss': total_loss,
            'bone_loss': bone_loss,
            'angle_loss': angle_loss
        }