"""
Gaussian Splatting Mapping

Creates 3D Gaussian splats from RGB-D point clouds.
"""

import numpy as np
from typing import List, Dict
from scipy.spatial import KDTree
import json


class GaussianSplat:
    """
    A single 3D Gaussian splat.
    
    Attributes:
        mean: 3D position [x, y, z]
        scale: Anisotropic scale [sx, sy, sz]
        rgb: Color [r, g, b] in [0, 1]
        opacity: Opacity in [0, 1]
    """
    
    def __init__(self,
                 mean: np.ndarray,
                 scale: np.ndarray,
                 rgb: np.ndarray,
                 opacity: float):
        self.mean = mean.copy()
        self.scale = scale.copy()
        self.rgb = rgb.copy()
        self.opacity = opacity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'mean': self.mean.tolist(),
            'scale': self.scale.tolist(),
            'rgb': self.rgb.tolist(),
            'opacity': float(self.opacity)
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'GaussianSplat':
        """Create from dictionary."""
        return GaussianSplat(
            mean=np.array(data['mean']),
            scale=np.array(data['scale']),
            rgb=np.array(data['rgb']),
            opacity=data['opacity']
        )


class GaussianSplatMapper:
    """
    Build 3D Gaussian splat map from RGB-D frames.
    
    Steps:
    1. Accumulate 3D points from all frames
    2. Voxel downsample
    3. Cluster nearby points
    4. For each cluster, estimate Gaussian parameters:
       - Mean: centroid
       - Scale: sqrt(eigenvalues) from covariance
       - RGB: average color
       - Opacity: based on point density
    5. Apply isotropic regularizer to avoid extreme elongation
    """
    
    def __init__(self,
                 voxel_size: float = 0.05,
                 cluster_radius: float = 0.10,
                 min_points_per_splat: int = 5,
                 isotropic_penalty_threshold: float = 3.0,
                 opacity_density_scale: float = 0.8):
        """
        Initialize Gaussian splat mapper.
        
        Args:
            voxel_size: Voxel size for downsampling (meters)
            cluster_radius: Radius for clustering points (meters)
            min_points_per_splat: Minimum points required for a splat
            isotropic_penalty_threshold: Threshold for scale ratio penalty
            opacity_density_scale: Scale factor for opacity from density
        """
        self.voxel_size = voxel_size
        self.cluster_radius = cluster_radius
        self.min_points_per_splat = min_points_per_splat
        self.isotropic_penalty_threshold = isotropic_penalty_threshold
        self.opacity_density_scale = opacity_density_scale
        
        self.accumulated_points = []
        self.accumulated_colors = []
    
    def add_frame(self,
                  rgb: np.ndarray,
                  depth: np.ndarray,
                  pose: np.ndarray,
                  intrinsics: np.ndarray):
        """
        Add an RGB-D frame to the map.
        
        Args:
            rgb: RGB image (H, W, 3) in [0, 255]
            depth: Depth image (H, W) in meters
            pose: Camera pose (4x4)
            intrinsics: Camera intrinsic matrix (3x3)
        """
        height, width = depth.shape
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Back-project to 3D
        for v in range(0, height, 2):  # Skip every other pixel for efficiency
            for u in range(0, width, 2):
                z = depth[v, u]
                
                # Skip invalid depth
                if z <= 0 or z > 10.0:
                    continue
                
                # Back-project to camera coordinates
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                # Transform to world coordinates
                point_cam = np.array([x, y, z, 1.0])
                point_world = pose @ point_cam
                
                # Get color
                color = rgb[v, u] / 255.0  # Normalize to [0, 1]
                
                self.accumulated_points.append(point_world[0:3])
                self.accumulated_colors.append(color)
    
    def build_splats(self) -> List[GaussianSplat]:
        """
        Build Gaussian splats from accumulated points.
        
        Returns:
            List of Gaussian splats
        """
        if len(self.accumulated_points) == 0:
            return []
        
        points = np.array(self.accumulated_points)
        colors = np.array(self.accumulated_colors)
        
        print(f"Accumulated {len(points)} points")
        
        # Voxel downsample
        points_down, colors_down = self._voxel_downsample(points, colors)
        print(f"After downsampling: {len(points_down)} points")
        
        # Cluster and create splats
        splats = self._cluster_and_create_splats(points_down, colors_down)
        print(f"Created {len(splats)} splats")
        
        return splats
    
    def _voxel_downsample(self,
                         points: np.ndarray,
                         colors: np.ndarray) -> tuple:
        """
        Voxel downsample points.
        
        Args:
            points: Nx3 array of points
            colors: Nx3 array of colors
        
        Returns:
            Downsampled points and colors
        """
        # Compute voxel indices
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
        
        # Create dictionary to accumulate points per voxel
        voxel_dict = {}
        
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_dict:
                voxel_dict[key] = {'points': [], 'colors': []}
            voxel_dict[key]['points'].append(points[i])
            voxel_dict[key]['colors'].append(colors[i])
        
        # Average points and colors per voxel
        downsampled_points = []
        downsampled_colors = []
        
        for voxel_data in voxel_dict.values():
            avg_point = np.mean(voxel_data['points'], axis=0)
            avg_color = np.mean(voxel_data['colors'], axis=0)
            downsampled_points.append(avg_point)
            downsampled_colors.append(avg_color)
        
        return np.array(downsampled_points), np.array(downsampled_colors)
    
    def _cluster_and_create_splats(self,
                                   points: np.ndarray,
                                   colors: np.ndarray) -> List[GaussianSplat]:
        """
        Cluster points and create Gaussian splats.
        
        Args:
            points: Nx3 array of points
            colors: Nx3 array of colors
        
        Returns:
            List of Gaussian splats
        """
        if len(points) == 0:
            return []
        
        # Build KD-tree
        tree = KDTree(points)
        
        # Track which points have been assigned to clusters
        assigned = np.zeros(len(points), dtype=bool)
        
        splats = []
        
        for i in range(len(points)):
            if assigned[i]:
                continue
            
            # Find neighbors within radius
            indices = tree.query_ball_point(points[i], self.cluster_radius)
            
            # Filter to unassigned neighbors
            indices = [idx for idx in indices if not assigned[idx]]
            
            if len(indices) < self.min_points_per_splat:
                assigned[i] = True
                continue
            
            # Create splat from cluster
            cluster_points = points[indices]
            cluster_colors = colors[indices]
            
            splat = self._create_splat_from_cluster(cluster_points, cluster_colors)
            if splat is not None:
                splats.append(splat)
            
            # Mark points as assigned
            for idx in indices:
                assigned[idx] = True
        
        return splats
    
    def _create_splat_from_cluster(self,
                                   points: np.ndarray,
                                   colors: np.ndarray) -> GaussianSplat:
        """
        Create a Gaussian splat from a cluster of points.
        
        Args:
            points: Nx3 array of cluster points
            colors: Nx3 array of cluster colors
        
        Returns:
            Gaussian splat or None
        """
        if len(points) < self.min_points_per_splat:
            return None
        
        # Mean position
        mean = np.mean(points, axis=0)
        
        # Covariance matrix
        centered = points - mean
        cov = (centered.T @ centered) / len(points)
        
        # Add small regularization to avoid singular matrices
        cov += np.eye(3) * 1e-6
        
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Scale as sqrt of eigenvalues
        scale = np.sqrt(np.abs(eigenvalues))
        
        # Apply isotropic penalty
        scale = self._apply_isotropic_penalty(scale)
        
        # Average color
        rgb = np.mean(colors, axis=0)
        rgb = np.clip(rgb, 0, 1)
        
        # Opacity based on point density
        density = len(points) / (self.cluster_radius ** 3)
        opacity = np.clip(density * self.opacity_density_scale, 0.1, 0.95)
        
        return GaussianSplat(mean, scale, rgb, opacity)
    
    def _apply_isotropic_penalty(self, scale: np.ndarray) -> np.ndarray:
        """
        Apply isotropic penalty to avoid extreme elongation.
        
        Args:
            scale: [sx, sy, sz]
        
        Returns:
            Regularized scale
        """
        max_scale = np.max(scale)
        min_scale = np.min(scale)
        
        # Compute ratio
        if min_scale > 1e-6:
            ratio = max_scale / min_scale
        else:
            ratio = 1.0
        
        # If ratio exceeds threshold, blend towards isotropic
        if ratio > self.isotropic_penalty_threshold:
            # Blend factor
            alpha = (ratio - self.isotropic_penalty_threshold) / ratio
            alpha = np.clip(alpha, 0, 0.5)  # Don't over-regularize
            
            # Isotropic scale (geometric mean)
            iso_scale = np.power(np.prod(scale), 1/3)
            
            # Blend
            scale = (1 - alpha) * scale + alpha * iso_scale
        
        # Ensure minimum scale
        scale = np.maximum(scale, self.voxel_size * 0.5)
        
        return scale


def save_splats(splats: List[GaussianSplat], filename: str):
    """
    Save splats to JSON file.
    
    Args:
        splats: List of Gaussian splats
        filename: Output file path
    """
    data = {
        'num_splats': len(splats),
        'splats': [splat.to_dict() for splat in splats]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_splats(filename: str) -> List[GaussianSplat]:
    """
    Load splats from JSON file.
    
    Args:
        filename: Input file path
    
    Returns:
        List of Gaussian splats
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    splats = [GaussianSplat.from_dict(splat_data) for splat_data in data['splats']]
    
    return splats

