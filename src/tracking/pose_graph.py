"""
Pose Graph Optimization

Simple sliding-window pose graph optimization to reduce drift.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Dict


class PoseGraphOptimizer:
    """
    Simple sliding-window pose graph optimizer.
    
    Maintains odometry edges between consecutive poses and optimizes
    a sliding window of recent poses to reduce drift.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize pose graph optimizer.
        
        Args:
            window_size: Number of recent poses to optimize
        """
        self.window_size = window_size
        self.poses = []  # List of 4x4 poses
        self.odometry_edges = []  # List of (i, j, T_rel) tuples
        self.timestamps = []  # List of timestamps
    
    def add_pose(self, pose: np.ndarray, timestamp: float, relative_transform: np.ndarray = None):
        """
        Add a new pose to the graph.
        
        Args:
            pose: 4x4 camera pose
            timestamp: Timestamp
            relative_transform: Relative transform from previous pose (4x4)
        """
        pose_idx = len(self.poses)
        self.poses.append(pose.copy())
        self.timestamps.append(timestamp)
        
        # Add odometry edge
        if pose_idx > 0 and relative_transform is not None:
            self.odometry_edges.append((pose_idx - 1, pose_idx, relative_transform.copy()))
    
    def optimize(self) -> List[np.ndarray]:
        """
        Optimize poses in the sliding window.
        
        Returns:
            Optimized poses
        """
        if len(self.poses) < 2:
            return self.poses
        
        # Determine window
        start_idx = max(0, len(self.poses) - self.window_size)
        end_idx = len(self.poses)
        
        # Extract poses in window
        window_poses = self.poses[start_idx:end_idx]
        
        # Extract relevant odometry edges
        window_edges = []
        for i, j, T_rel in self.odometry_edges:
            if start_idx <= i < end_idx and start_idx <= j < end_idx:
                # Adjust indices to window coordinates
                window_edges.append((i - start_idx, j - start_idx, T_rel))
        
        if len(window_edges) == 0:
            return self.poses
        
        # Optimize window
        optimized_window = self._optimize_window(window_poses, window_edges)
        
        # Update poses
        for i, opt_pose in enumerate(optimized_window):
            self.poses[start_idx + i] = opt_pose
        
        return self.poses
    
    def _optimize_window(self,
                        poses: List[np.ndarray],
                        edges: List[Tuple[int, int, np.ndarray]]) -> List[np.ndarray]:
        """
        Optimize a window of poses given odometry constraints.
        
        Args:
            poses: List of 4x4 poses
            edges: List of (i, j, T_rel) odometry constraints
        
        Returns:
            Optimized poses
        """
        n = len(poses)
        
        # Convert poses to parameter vector (tx, ty, tz, rx, ry, rz per pose)
        x0 = self._poses_to_vector(poses)
        
        # Fix first pose (anchor)
        def residuals(x):
            # Reconstruct poses
            current_poses = self._vector_to_poses(x, n)
            
            res = []
            
            # Odometry edge residuals
            for i, j, T_rel in edges:
                T_i = current_poses[i]
                T_j = current_poses[j]
                
                # Predicted relative transform
                T_pred = np.linalg.inv(T_i) @ T_j
                
                # Error between predicted and measured
                T_error = np.linalg.inv(T_rel) @ T_pred
                
                # Extract translation and rotation errors
                t_error = T_error[0:3, 3]
                R_error = T_error[0:3, 0:3]
                
                # Rotation error as angle-axis
                angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
                axis = np.array([
                    R_error[2, 1] - R_error[1, 2],
                    R_error[0, 2] - R_error[2, 0],
                    R_error[1, 0] - R_error[0, 1]
                ])
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                else:
                    axis = np.array([0, 0, 0])
                r_error = angle * axis
                
                # Combine errors
                res.extend(t_error * 10)  # Weight translation more
                res.extend(r_error)
            
            # Anchor first pose
            res.extend((x[0:6] - x0[0:6]) * 100)  # Strong weight on anchor
            
            return np.array(res)
        
        # Optimize
        result = least_squares(residuals, x0, method='lm', max_nfev=50)
        
        # Convert back to poses
        optimized_poses = self._vector_to_poses(result.x, n)
        
        return optimized_poses
    
    @staticmethod
    def _poses_to_vector(poses: List[np.ndarray]) -> np.ndarray:
        """Convert list of 4x4 poses to parameter vector."""
        params = []
        for pose in poses:
            # Translation
            t = pose[0:3, 3]
            
            # Rotation as axis-angle
            R = pose[0:3, 0:3]
            angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ])
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
            else:
                axis = np.array([0, 0, 1])  # Default to z-axis
            r = angle * axis
            
            params.extend(t)
            params.extend(r)
        
        return np.array(params)
    
    @staticmethod
    def _vector_to_poses(x: np.ndarray, n: int) -> List[np.ndarray]:
        """Convert parameter vector to list of 4x4 poses."""
        poses = []
        for i in range(n):
            # Extract parameters
            t = x[i*6:i*6+3]
            r = x[i*6+3:i*6+6]
            
            # Convert axis-angle to rotation matrix
            angle = np.linalg.norm(r)
            if angle > 1e-6:
                axis = r / angle
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            else:
                R = np.eye(3)
            
            # Build 4x4 pose
            pose = np.eye(4)
            pose[0:3, 0:3] = R
            pose[0:3, 3] = t
            
            poses.append(pose)
        
        return poses
    
    def get_poses(self) -> List[np.ndarray]:
        """Get all poses."""
        return self.poses
    
    def get_timestamps(self) -> List[float]:
        """Get all timestamps."""
        return self.timestamps


def compute_relative_transform(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Compute relative transformation from pose1 to pose2.
    
    Args:
        pose1: First pose (4x4)
        pose2: Second pose (4x4)
    
    Returns:
        Relative transformation (4x4)
    """
    return np.linalg.inv(pose1) @ pose2


def interpolate_pose(pose1: np.ndarray, pose2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two poses.
    
    Args:
        pose1: First pose (4x4)
        pose2: Second pose (4x4)
        alpha: Interpolation factor [0, 1]
    
    Returns:
        Interpolated pose (4x4)
    """
    # Interpolate translation
    t1 = pose1[0:3, 3]
    t2 = pose2[0:3, 3]
    t = (1 - alpha) * t1 + alpha * t2
    
    # Interpolate rotation (simple linear blend, not SLERP)
    R1 = pose1[0:3, 0:3]
    R2 = pose2[0:3, 0:3]
    R = (1 - alpha) * R1 + alpha * R2
    
    # Orthogonalize (nearest rotation matrix)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Build interpolated pose
    pose = np.eye(4)
    pose[0:3, 0:3] = R
    pose[0:3, 3] = t
    
    return pose

