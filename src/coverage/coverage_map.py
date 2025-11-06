"""
Coverage Map

Track observation counts per grid cell and compute coverage completeness.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json


class CoverageMap:
    """
    2D coverage map that tracks observation counts.
    
    Maintains a grid where each cell counts how many times it has been observed
    across all frames.
    """
    
    def __init__(self,
                 resolution: float = 0.05,
                 min_observations: int = 3):
        """
        Initialize coverage map.
        
        Args:
            resolution: Grid resolution in meters
            min_observations: Minimum observations for "complete" coverage
        """
        self.resolution = resolution
        self.min_observations = min_observations
        self.grid = None
        self.origin = None
        self.shape = None
    
    def initialize_from_bounds(self,
                               min_xy: np.ndarray,
                               max_xy: np.ndarray):
        """
        Initialize grid from spatial bounds.
        
        Args:
            min_xy: Minimum [x, y] coordinates
            max_xy: Maximum [x, y] coordinates
        """
        # Add padding (reduced from 1.0 to avoid excessive grid size)
        padding = 0.5
        min_xy = min_xy - padding
        max_xy = max_xy + padding
        
        # Grid dimensions
        grid_size = max_xy - min_xy
        grid_shape = np.ceil(grid_size / self.resolution).astype(int)
        
        self.origin = min_xy
        self.shape = (grid_shape[1], grid_shape[0])  # (height, width)
        self.grid = np.zeros(self.shape, dtype=np.int32)
    
    def add_observation(self,
                       rgb: np.ndarray,
                       depth: np.ndarray,
                       pose: np.ndarray,
                       intrinsics: np.ndarray,
                       floor_height: float = 0.0):
        """
        Add observations from a frame.
        
        Args:
            rgb: RGB image (not used, for consistency)
            depth: Depth image
            pose: Camera pose (4x4)
            intrinsics: Camera intrinsic matrix
            floor_height: Floor Z coordinate
        """
        if self.grid is None:
            raise ValueError("Coverage map not initialized")
        
        height, width = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Sample pixels (improved sampling for better coverage)
        step = 2  # Reduced from 5 to sample 25% of pixels instead of 4%
        for v in range(0, height, step):
            for u in range(0, width, step):
                z = depth[v, u]
                
                if z <= 0 or z > 15.0:  # Increased from 10.0 to capture more distant points
                    continue
                
                # Back-project to 3D
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                point_cam = np.array([x, y, z, 1.0])
                point_world = pose @ point_cam
                
                # Project to 2D grid (only if near floor)
                # More lenient height filter to capture more floor observations
                point_height = point_world[2] - floor_height
                if -0.3 <= point_height <= 1.5:  # Expanded from -0.2 to 1.0 for better coverage
                    idx = self._world_to_grid(point_world[0:2])
                    if self._is_valid_index(idx):
                        self.grid[idx[1], idx[0]] += 1
    
    def _world_to_grid(self, point: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        idx = ((point - self.origin) / self.resolution).astype(int)
        return idx[0], idx[1]
    
    def _is_valid_index(self, idx: Tuple[int, int]) -> bool:
        """Check if grid index is valid."""
        return 0 <= idx[0] < self.shape[1] and 0 <= idx[1] < self.shape[0]
    
    def compute_completeness(self) -> float:
        """
        Compute coverage completeness.
        
        Returns:
            Fraction of cells with >= min_observations [0, 1]
        """
        if self.grid is None:
            return 0.0
        
        total_cells = np.prod(self.grid.shape)
        complete_cells = np.sum(self.grid >= self.min_observations)
        
        completeness = complete_cells / total_cells
        
        return completeness
    
    def get_low_coverage_regions(self, threshold: int = 1) -> List[np.ndarray]:
        """
        Identify regions with low coverage.
        
        Args:
            threshold: Maximum observation count for "low coverage"
        
        Returns:
            List of world coordinates of low-coverage cells
        """
        if self.grid is None:
            return []
        
        low_coverage_indices = np.argwhere(self.grid <= threshold)
        
        # Convert to world coordinates
        low_coverage_world = []
        for idx in low_coverage_indices:
            y, x = idx
            world_coord = self.origin + np.array([x, y]) * self.resolution
            low_coverage_world.append(world_coord)
        
        # Subsample if too many
        if len(low_coverage_world) > 100:
            indices = np.random.choice(len(low_coverage_world), 100, replace=False)
            low_coverage_world = [low_coverage_world[i] for i in indices]
        
        return low_coverage_world
    
    def save_heatmap(self, filename: str):
        """
        Save coverage heatmap visualization.
        
        Args:
            filename: Output filename
        """
        if self.grid is None:
            return
        
        height, width = self.shape
        extent = [
            self.origin[0],
            self.origin[0] + width * self.resolution,
            self.origin[1],
            self.origin[1] + height * self.resolution
        ]
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        im = plt.imshow(
            self.grid,
            cmap='YlOrRd',
            origin='lower',
            extent=extent,
            interpolation='nearest'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, label='Observation Count')
        
        # Mark low-coverage regions
        low_coverage = self.get_low_coverage_regions(threshold=self.min_observations - 1)
        if len(low_coverage) > 0:
            low_coverage = np.array(low_coverage)
            plt.scatter(
                low_coverage[:, 0],
                low_coverage[:, 1],
                c='blue',
                marker='x',
                s=20,
                alpha=0.5,
                label='Low Coverage'
            )
        
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title(f'Coverage Heatmap (Completeness: {self.compute_completeness():.1%})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
    
    def get_statistics(self) -> dict:
        """
        Get coverage statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.grid is None:
            return {}
        
        return {
            'completeness': float(self.compute_completeness()),
            'min_observations': int(np.min(self.grid)),
            'max_observations': int(np.max(self.grid)),
            'mean_observations': float(np.mean(self.grid)),
            'median_observations': float(np.median(self.grid)),
            'total_cells': int(np.prod(self.grid.shape)),
            'observed_cells': int(np.sum(self.grid > 0)),
            'complete_cells': int(np.sum(self.grid >= self.min_observations))
        }


def compute_trajectory_metrics(poses: np.ndarray,
                               timestamps: List[float],
                               gt_poses: Optional[np.ndarray] = None,
                               gt_timestamps: Optional[List[float]] = None) -> dict:
    """
    Compute trajectory metrics.
    
    Args:
        poses: Nx4x4 estimated poses
        timestamps: Timestamps
        gt_poses: Ground truth poses (optional)
        gt_timestamps: Ground truth timestamps (optional)
    
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Trajectory length
    trajectory_length = 0.0
    for i in range(1, len(poses)):
        delta = poses[i][0:3, 3] - poses[i-1][0:3, 3]
        trajectory_length += np.linalg.norm(delta)
    
    metrics['trajectory_length'] = float(trajectory_length)
    metrics['num_frames'] = len(poses)
    
    # ATE (Absolute Trajectory Error) if ground truth available
    if gt_poses is not None and len(gt_poses) > 0:
        ate_rmse = compute_ate_rmse(poses, timestamps, gt_poses, gt_timestamps)
        metrics['ate_rmse'] = float(ate_rmse) if ate_rmse is not None else None
    else:
        metrics['ate_rmse'] = None
    
    return metrics


def compute_ate_rmse(est_poses: np.ndarray,
                     est_timestamps: List[float],
                     gt_poses: np.ndarray,
                     gt_timestamps: List[float]) -> Optional[float]:
    """
    Compute Absolute Trajectory Error (ATE) RMSE.
    
    Args:
        est_poses: Estimated poses (Nx4x4)
        est_timestamps: Estimated timestamps
        gt_poses: Ground truth poses (Mx4x4)
        gt_timestamps: Ground truth timestamps
    
    Returns:
        ATE RMSE in meters or None
    """
    if len(est_poses) == 0 or len(gt_poses) == 0:
        return None
    
    # Associate timestamps
    errors = []
    
    for i, ts in enumerate(est_timestamps):
        # Find closest ground truth timestamp
        idx = np.argmin(np.abs(np.array(gt_timestamps) - ts))
        dt = abs(gt_timestamps[idx] - ts)
        
        # Only use if timestamps are close
        if dt > 0.05:
            continue
        
        # Translation error
        est_trans = est_poses[i][0:3, 3]
        gt_trans = gt_poses[idx][0:3, 3]
        error = np.linalg.norm(est_trans - gt_trans)
        errors.append(error)
    
    if len(errors) == 0:
        return None
    
    # RMSE
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    
    return rmse


def save_metrics_json(metrics: dict, filename: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

