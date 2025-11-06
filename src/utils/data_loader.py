"""
TUM RGB-D Dataset Loader

Loads RGB-D sequences in TUM format with timestamps, camera intrinsics,
and optional ground truth poses.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import cv2


class TUMDataLoader:
    """
    Loader for TUM RGB-D dataset format.
    
    Expected directory structure:
        data_dir/
            rgb.txt           # timestamp filename
            depth.txt         # timestamp filename
            groundtruth.txt   # timestamp tx ty tz qx qy qz qw (optional)
            rgb/              # RGB images
            depth/            # Depth images
    """
    
    def __init__(self, data_dir: str, frame_skip: int = 1):
        """
        Initialize TUM data loader.
        
        Args:
            data_dir: Path to TUM dataset directory
            frame_skip: Process every Nth frame (1 = all frames)
        """
        self.data_dir = Path(data_dir)
        self.frame_skip = frame_skip
        
        # Load associations
        self.rgb_files, self.rgb_timestamps = self._load_file_list('rgb.txt')
        self.depth_files, self.depth_timestamps = self._load_file_list('depth.txt')
        
        # Associate RGB and depth by timestamp
        self.associations = self._associate_frames()
        
        # Load ground truth if available
        self.has_groundtruth = (self.data_dir / 'groundtruth.txt').exists()
        if self.has_groundtruth:
            self.gt_poses, self.gt_timestamps = self._load_groundtruth()
        else:
            self.gt_poses = None
            self.gt_timestamps = None
        
        # Load camera intrinsics (from standard TUM Freiburg1 camera)
        self.intrinsics = self._load_intrinsics()
        
        print(f"Loaded {len(self.associations)} frame pairs")
        if self.has_groundtruth:
            print(f"Ground truth poses available: {len(self.gt_poses)}")
    
    def _load_file_list(self, filename: str) -> Tuple[List[str], List[float]]:
        """Load file list from rgb.txt or depth.txt."""
        filepath = self.data_dir / filename
        files = []
        timestamps = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    timestamps.append(float(parts[0]))
                    files.append(parts[1])
        
        return files, timestamps
    
    def _associate_frames(self, max_dt: float = 0.02) -> List[Tuple[int, int]]:
        """
        Associate RGB and depth frames by timestamp.
        
        Args:
            max_dt: Maximum time difference (seconds) for association
        
        Returns:
            List of (rgb_idx, depth_idx) pairs
        """
        associations = []
        
        for i, rgb_ts in enumerate(self.rgb_timestamps):
            # Find closest depth timestamp
            best_j = None
            best_dt = float('inf')
            
            for j, depth_ts in enumerate(self.depth_timestamps):
                dt = abs(rgb_ts - depth_ts)
                if dt < best_dt:
                    best_dt = dt
                    best_j = j
            
            if best_j is not None and best_dt < max_dt:
                associations.append((i, best_j))
        
        return associations
    
    def _load_groundtruth(self) -> Tuple[np.ndarray, List[float]]:
        """
        Load ground truth poses in TUM format.
        
        Returns:
            poses: Nx7 array (tx, ty, tz, qx, qy, qz, qw)
            timestamps: List of timestamps
        """
        filepath = self.data_dir / 'groundtruth.txt'
        poses = []
        timestamps = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    timestamp = float(parts[0])
                    pose = [float(x) for x in parts[1:8]]  # tx, ty, tz, qx, qy, qz, qw
                    timestamps.append(timestamp)
                    poses.append(pose)
        
        return np.array(poses), timestamps
    
    def _load_intrinsics(self) -> Dict[str, float]:
        """
        Load camera intrinsics.
        
        Uses default TUM Freiburg1 parameters if no intrinsics file found.
        """
        # Check for intrinsics file
        intrinsics_file = self.data_dir / 'camera.txt'
        
        if intrinsics_file.exists():
            with open(intrinsics_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Expected format: fx fy cx cy
                    parts = line.split()
                    if len(parts) >= 4:
                        return {
                            'fx': float(parts[0]),
                            'fy': float(parts[1]),
                            'cx': float(parts[2]),
                            'cy': float(parts[3]),
                            'width': 640,
                            'height': 480
                        }
        
        # Default TUM Freiburg1 intrinsics
        return {
            'fx': 517.3,
            'fy': 516.5,
            'cx': 318.6,
            'cy': 255.3,
            'width': 640,
            'height': 480
        }
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix."""
        K = np.eye(3)
        K[0, 0] = self.intrinsics['fx']
        K[1, 1] = self.intrinsics['fy']
        K[0, 2] = self.intrinsics['cx']
        K[1, 2] = self.intrinsics['cy']
        return K
    
    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.associations) // self.frame_skip
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get frame data by index.
        
        Args:
            idx: Frame index
        
        Returns:
            Dictionary with keys:
                - rgb: RGB image (H, W, 3)
                - depth: Depth image (H, W) in meters
                - timestamp: Frame timestamp
                - rgb_timestamp: RGB timestamp
                - depth_timestamp: Depth timestamp
                - pose_gt: Ground truth pose (4x4) if available, else None
        """
        # Apply frame skip
        actual_idx = idx * self.frame_skip
        if actual_idx >= len(self.associations):
            raise IndexError(f"Index {idx} out of range")
        
        rgb_idx, depth_idx = self.associations[actual_idx]
        
        # Load RGB image
        rgb_path = self.data_dir / self.rgb_files[rgb_idx]
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load depth image
        depth_path = self.data_dir / self.depth_files[depth_idx]
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        # Convert to meters (TUM depth is in millimeters for 16-bit images)
        depth = depth.astype(np.float32) / 5000.0
        
        # Get timestamps
        rgb_ts = self.rgb_timestamps[rgb_idx]
        depth_ts = self.depth_timestamps[depth_idx]
        timestamp = (rgb_ts + depth_ts) / 2.0  # Average timestamp
        
        # Get ground truth pose if available
        pose_gt = None
        if self.has_groundtruth:
            pose_gt = self._get_closest_pose(timestamp)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'timestamp': timestamp,
            'rgb_timestamp': rgb_ts,
            'depth_timestamp': depth_ts,
            'pose_gt': pose_gt
        }
    
    def _get_closest_pose(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Get closest ground truth pose for a given timestamp.
        
        Returns:
            4x4 transformation matrix or None
        """
        if not self.has_groundtruth:
            return None
        
        # Find closest timestamp
        idx = np.argmin(np.abs(np.array(self.gt_timestamps) - timestamp))
        dt = abs(self.gt_timestamps[idx] - timestamp)
        
        # Only return pose if within 0.02 seconds
        if dt > 0.02:
            return None
        
        # Convert quaternion + translation to 4x4 matrix
        pose_data = self.gt_poses[idx]
        tx, ty, tz = pose_data[0:3]
        qx, qy, qz, qw = pose_data[3:7]
        
        # Quaternion to rotation matrix
        R = self._quaternion_to_matrix(qx, qy, qz, qw)
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [tx, ty, tz]
        
        return T
    
    @staticmethod
    def _quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        # Normalize quaternion
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
    
    def get_all_groundtruth_poses(self) -> Optional[np.ndarray]:
        """
        Get all ground truth poses as 4x4 matrices.
        
        Returns:
            Nx4x4 array of transformation matrices or None
        """
        if not self.has_groundtruth:
            return None
        
        poses = []
        for pose_data in self.gt_poses:
            tx, ty, tz = pose_data[0:3]
            qx, qy, qz, qw = pose_data[3:7]
            
            R = self._quaternion_to_matrix(qx, qy, qz, qw)
            
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = [tx, ty, tz]
            
            poses.append(T)
        
        return np.array(poses)


def save_trajectory_tum(filename: str, timestamps: List[float], poses: np.ndarray):
    """
    Save trajectory in TUM format.
    
    Args:
        filename: Output file path
        timestamps: List of timestamps
        poses: Nx4x4 transformation matrices
    """
    with open(filename, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        for ts, pose in zip(timestamps, poses):
            # Extract translation
            tx, ty, tz = pose[0:3, 3]
            
            # Extract rotation and convert to quaternion
            R = pose[0:3, 0:3]
            qw, qx, qy, qz = matrix_to_quaternion(R)
            
            f.write(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                   f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert rotation matrix to quaternion (qw, qx, qy, qz).
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        (qw, qx, qy, qz)
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return qw, qx, qy, qz

