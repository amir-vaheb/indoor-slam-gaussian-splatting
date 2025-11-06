"""
RGB-D ICP Tracking

Estimates camera pose using Iterative Closest Point (ICP) registration.
"""

import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Dict


class ICPTracker:
    """
    RGB-D tracking using ICP (point-to-plane or colored ICP).
    
    Steps:
    1. Convert RGB-D to point cloud
    2. Align current point cloud with previous using ICP
    3. Return relative transformation
    """
    
    def __init__(self,
                 icp_type: str = "point_to_plane",
                 max_correspondence_distance: float = 0.05,
                 voxel_size: float = 0.02,
                 use_color: bool = True):
        """
        Initialize ICP tracker.
        
        Args:
            icp_type: "point_to_point" or "point_to_plane"
            max_correspondence_distance: Maximum distance for point correspondences (meters)
            voxel_size: Voxel size for downsampling (meters)
            use_color: Whether to use colored ICP
        """
        self.icp_type = icp_type
        self.max_correspondence_distance = max_correspondence_distance
        self.voxel_size = voxel_size
        self.use_color = use_color
        
        # Previous point cloud
        self.prev_pcd = None
        self.prev_pose = np.eye(4)
    
    def rgbd_to_pointcloud(self,
                          rgb: np.ndarray,
                          depth: np.ndarray,
                          intrinsics: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Convert RGB-D image to Open3D point cloud.
        
        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) in meters
            intrinsics: 3x3 camera intrinsic matrix
        
        Returns:
            Open3D point cloud
        """
        height, width = depth.shape
        
        # Create Open3D RGB-D image
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # Convert to mm
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        )
        
        # Create intrinsic object
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic_o3d
        )
        
        return pcd
    
    def preprocess_pointcloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Preprocess point cloud (downsample, estimate normals).
        
        Args:
            pcd: Input point cloud
        
        Returns:
            Processed point cloud
        """
        # Downsample
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2,
                max_nn=30
            )
        )
        
        return pcd_down
    
    def track(self,
              rgb: np.ndarray,
              depth: np.ndarray,
              intrinsics: np.ndarray,
              prev_pose: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray, Dict]:
        """
        Track camera pose from RGB-D frame using ICP.
        
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
        # Convert to point cloud
        pcd = self.rgbd_to_pointcloud(rgb, depth, intrinsics)
        
        # Preprocess
        pcd_processed = self.preprocess_pointcloud(pcd)
        
        # Initialize if first frame
        if self.prev_pcd is None:
            self.prev_pcd = pcd_processed
            if prev_pose is not None:
                self.prev_pose = prev_pose
            
            info = {
                'num_points': len(pcd_processed.points),
                'fitness': 1.0,
                'inlier_rmse': 0.0
            }
            return True, self.prev_pose.copy(), info
        
        # Initial transformation guess (use previous relative motion or identity)
        init_transform = np.eye(4)
        
        # Run ICP
        if self.use_color and pcd_processed.has_colors():
            # Colored ICP
            result = o3d.pipelines.registration.registration_colored_icp(
                source=pcd_processed,
                target=self.prev_pcd,
                max_correspondence_distance=self.max_correspondence_distance,
                init=init_transform,
                estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=30
                )
            )
        else:
            # Standard ICP
            if self.icp_type == "point_to_plane":
                result = o3d.pipelines.registration.registration_icp(
                    source=pcd_processed,
                    target=self.prev_pcd,
                    max_correspondence_distance=self.max_correspondence_distance,
                    init=init_transform,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=30
                    )
                )
            else:  # point_to_point
                result = o3d.pipelines.registration.registration_icp(
                    source=pcd_processed,
                    target=self.prev_pcd,
                    max_correspondence_distance=self.max_correspondence_distance,
                    init=init_transform,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=30
                    )
                )
        
        # Check if ICP succeeded
        if result.fitness < 0.3:
            # Low fitness, tracking may have failed
            success = False
            pose = self.prev_pose.copy()
        else:
            success = True
            # Relative transformation from prev to current
            T_rel = result.transformation
            
            # Compose with previous pose
            pose = self.prev_pose @ T_rel
            
            # Update state
            self.prev_pose = pose.copy()
        
        # Update previous point cloud
        self.prev_pcd = pcd_processed
        
        info = {
            'num_points': len(pcd_processed.points),
            'fitness': result.fitness,
            'inlier_rmse': result.inlier_rmse,
            'correspondence_set': len(result.correspondence_set)
        }
        
        return success, pose, info
    
    def reset(self):
        """Reset tracker state."""
        self.prev_pcd = None
        self.prev_pose = np.eye(4)


def estimate_relative_pose_icp(
    prev_rgb: np.ndarray,
    prev_depth: np.ndarray,
    curr_rgb: np.ndarray,
    curr_depth: np.ndarray,
    intrinsics: np.ndarray,
    max_correspondence_distance: float = 0.05
) -> Tuple[bool, np.ndarray, Dict]:
    """
    Estimate relative pose between two frames using ICP.
    
    Args:
        prev_rgb: Previous RGB frame
        prev_depth: Previous depth frame
        curr_rgb: Current RGB frame
        curr_depth: Current depth frame
        intrinsics: Camera intrinsic matrix
        max_correspondence_distance: Max distance for correspondences
    
    Returns:
        success: Whether estimation succeeded
        T_rel: Relative transformation (4x4)
        info: Tracking statistics
    """
    tracker = ICPTracker(max_correspondence_distance=max_correspondence_distance)
    
    # Process first frame
    tracker.track(prev_rgb, prev_depth, intrinsics)
    
    # Process second frame
    success, pose, info = tracker.track(curr_rgb, curr_depth, intrinsics)
    
    # Extract relative transformation
    T_rel = pose
    
    return success, T_rel, info

