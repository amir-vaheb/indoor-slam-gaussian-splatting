"""
Feature-based PnP + RANSAC Tracking

Estimates camera pose using feature matching and PnP with RANSAC.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict


class FeaturePnPTracker:
    """
    Feature-based RGB-D tracking using PnP + RANSAC.
    
    Steps:
    1. Extract features (ORB/AKAZE) from RGB frames
    2. Match features between consecutive frames
    3. Back-project 2D features to 3D using depth
    4. Estimate relative pose using PnP + RANSAC
    """
    
    def __init__(self, 
                 feature_type: str = "ORB",
                 max_features: int = 2000,
                 match_ratio: float = 0.75,
                 ransac_threshold: float = 0.01):
        """
        Initialize feature-based tracker.
        
        Args:
            feature_type: "ORB" or "AKAZE"
            max_features: Maximum number of features to detect
            match_ratio: Ratio test threshold for feature matching
            ransac_threshold: RANSAC inlier threshold (meters)
        """
        self.feature_type = feature_type
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        
        # Create feature detector
        if feature_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif feature_type == "AKAZE":
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Create matcher
        if feature_type == "ORB":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_depth = None
    
    def track(self, 
              rgb: np.ndarray, 
              depth: np.ndarray,
              intrinsics: np.ndarray,
              prev_pose: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray, Dict]:
        """
        Track camera pose from RGB-D frame.
        
        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth image (H, W) in meters
            intrinsics: 3x3 camera intrinsic matrix
            prev_pose: Previous camera pose (4x4), optional
        
        Returns:
            success: Whether tracking succeeded
            pose: Estimated camera pose (4x4)
            info: Dictionary with tracking statistics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        # Initialize if first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_depth = depth
            
            # Return identity pose for first frame
            pose = np.eye(4) if prev_pose is None else prev_pose
            info = {'num_features': len(keypoints), 'num_matches': 0, 'num_inliers': 0}
            return True, pose, info
        
        # Match features with previous frame
        if descriptors is None or self.prev_descriptors is None:
            # No features detected, return previous pose
            pose = np.eye(4) if prev_pose is None else prev_pose
            info = {'num_features': len(keypoints), 'num_matches': 0, 'num_inliers': 0}
            return False, pose, info
        
        # Find matches using KNN
        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            # Not enough matches
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_depth = depth
            
            pose = np.eye(4) if prev_pose is None else prev_pose
            info = {'num_features': len(keypoints), 'num_matches': len(good_matches), 'num_inliers': 0}
            return False, pose, info
        
        # Back-project matched features to 3D using depth
        points_3d = []
        points_2d = []
        
        for match in good_matches:
            # Previous frame keypoint (3D)
            prev_kp = self.prev_keypoints[match.queryIdx]
            px, py = int(prev_kp.pt[0]), int(prev_kp.pt[1])
            
            # Check bounds
            if px < 0 or px >= self.prev_depth.shape[1] or py < 0 or py >= self.prev_depth.shape[0]:
                continue
            
            z = self.prev_depth[py, px]
            
            # Skip invalid depth
            if z <= 0 or z > 10.0:
                continue
            
            # Back-project to 3D
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            x = (px - cx) * z / fx
            y = (py - cy) * z / fy
            points_3d.append([x, y, z])
            
            # Current frame keypoint (2D)
            curr_kp = keypoints[match.trainIdx]
            points_2d.append(curr_kp.pt)
        
        if len(points_3d) < 10:
            # Not enough valid 3D-2D correspondences
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_depth = depth
            
            pose = np.eye(4) if prev_pose is None else prev_pose
            info = {'num_features': len(keypoints), 'num_matches': len(good_matches), 'num_inliers': 0}
            return False, pose, info
        
        points_3d = np.array(points_3d, dtype=np.float64)
        points_2d = np.array(points_2d, dtype=np.float64)
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            intrinsics,
            None,
            reprojectionError=self.ransac_threshold * intrinsics[0, 0],  # Convert to pixels
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < 10:
            # PnP failed
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_depth = depth
            
            pose = np.eye(4) if prev_pose is None else prev_pose
            info = {'num_features': len(keypoints), 'num_matches': len(good_matches), 'num_inliers': 0}
            return False, pose, info
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Build transformation matrix (this is camera motion from prev to curr)
        T_rel = np.eye(4)
        T_rel[0:3, 0:3] = R
        T_rel[0:3, 3] = tvec.flatten()
        
        # Compose with previous pose
        if prev_pose is None:
            pose = T_rel
        else:
            pose = prev_pose @ T_rel
        
        # Update previous frame data
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_depth = depth
        
        info = {
            'num_features': len(keypoints),
            'num_matches': len(good_matches),
            'num_inliers': len(inliers),
            'inlier_ratio': len(inliers) / len(points_3d)
        }
        
        return True, pose, info
    
    def reset(self):
        """Reset tracker state."""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_depth = None


def estimate_relative_pose_pnp(
    prev_rgb: np.ndarray,
    prev_depth: np.ndarray,
    curr_rgb: np.ndarray,
    intrinsics: np.ndarray,
    feature_type: str = "ORB",
    max_features: int = 2000
) -> Tuple[bool, np.ndarray, Dict]:
    """
    Estimate relative pose between two frames using feature PnP.
    
    Args:
        prev_rgb: Previous RGB frame
        prev_depth: Previous depth frame
        curr_rgb: Current RGB frame
        intrinsics: Camera intrinsic matrix
        feature_type: Feature detector type
        max_features: Maximum features to detect
    
    Returns:
        success: Whether estimation succeeded
        T_rel: Relative transformation (4x4)
        info: Tracking statistics
    """
    tracker = FeaturePnPTracker(feature_type=feature_type, max_features=max_features)
    
    # Process first frame
    tracker.track(prev_rgb, prev_depth, intrinsics)
    
    # Process second frame
    success, T_rel, info = tracker.track(curr_rgb, prev_depth, intrinsics)
    
    return success, T_rel, info

