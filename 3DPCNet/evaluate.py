"""
Evaluation utilities for pose canonicalization with similarity transform metrics.
"""

import numpy as np
import torch


def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c


def calculate_error(preds, gts):
    """
    Compute MPJPE and PA-MPJPE given predictions and ground-truths.
    
    Args:
        preds: (N, J, 3) predicted poses
        gts: (N, J, 3) ground truth poses
        
    Returns:
        mpjpe: Mean Per Joint Position Error
        pampjpe: Procrustes Aligned Mean Per Joint Position Error
    """
    N = preds.shape[0]
    num_joints = preds.shape[1]

    # MPJPE: direct Euclidean distance
    mpjpe = np.mean(np.sqrt(np.sum(np.square(preds - gts), axis=2)))

    # PA-MPJPE: after similarity transform alignment
    pampjpe = np.zeros([N, num_joints])

    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]
        _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred = (b * frame_pred.dot(T)) + c
        pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))

    pampjpe = np.mean(pampjpe)

    return mpjpe, pampjpe


def evaluate_pose_canonicalization(pred_canonical, gt_canonical, pred_rotation, gt_rotation):
    """
    Evaluate pose canonicalization model with multiple metrics.
    
    Args:
        pred_canonical: (N, J, 3) predicted canonical poses
        gt_canonical: (N, J, 3) ground truth canonical poses  
        pred_rotation: (N, 3, 3) predicted rotation matrices
        gt_rotation: (N, 3, 3) ground truth rotation matrices
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Convert to numpy for evaluation
    if isinstance(pred_canonical, torch.Tensor):
        pred_canonical = pred_canonical.detach().cpu().numpy()
    if isinstance(gt_canonical, torch.Tensor):
        gt_canonical = gt_canonical.detach().cpu().numpy()
    if isinstance(pred_rotation, torch.Tensor):
        pred_rotation = pred_rotation.detach().cpu().numpy()
    if isinstance(gt_rotation, torch.Tensor):
        gt_rotation = gt_rotation.detach().cpu().numpy()
    
    # Pose metrics
    mpjpe, pampjpe = calculate_error(pred_canonical, gt_canonical)
    
    # Rotation metrics
    rotation_error = np.mean(np.linalg.norm(pred_rotation - gt_rotation, axis=(1, 2)))
    
    # Convert rotation error to degrees
    rotation_error_deg = rotation_error * 180.0 / np.pi
    
    return {
        'mpjpe': mpjpe,
        'pampjpe': pampjpe, 
        'rotation_error': rotation_error,
        'rotation_error_deg': rotation_error_deg
    }
