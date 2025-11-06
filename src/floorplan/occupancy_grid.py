"""
Occupancy Grid

Project 3D points to 2D occupancy grid for floorplan extraction.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class OccupancyGrid:
    """
    2D occupancy grid for floorplan generation.
    
    Projects 3D points (typically wall points) to a 2D grid and
    maintains occupancy information.
    """
    
    def __init__(self, resolution: float = 0.05):
        """
        Initialize occupancy grid.
        
        Args:
            resolution: Grid cell size in meters
        """
        self.resolution = resolution
        self.grid = None
        self.origin = None  # (x_min, y_min)
        self.shape = None  # (height, width)
    
    def build_from_points(self, points: np.ndarray) -> np.ndarray:
        """
        Build occupancy grid from 3D points.
        
        Args:
            points: Nx3 array of 3D points (typically wall points)
        
        Returns:
            Occupancy grid (H, W) with values in [0, 1]
        """
        if len(points) == 0:
            return np.zeros((100, 100))
        
        # Project to 2D (ignore Z)
        points_2d = points[:, 0:2]
        
        # Determine grid bounds
        min_xy = np.min(points_2d, axis=0)
        max_xy = np.max(points_2d, axis=0)
        
        # Add padding
        padding = 1.0  # meters
        min_xy -= padding
        max_xy += padding
        
        # Grid dimensions
        grid_size = max_xy - min_xy
        grid_shape = np.ceil(grid_size / self.resolution).astype(int)
        
        self.origin = min_xy
        self.shape = (grid_shape[1], grid_shape[0])  # (height, width)
        
        # Initialize grid
        self.grid = np.zeros(self.shape, dtype=np.float32)
        
        # Project points to grid
        for point_2d in points_2d:
            idx = self._world_to_grid(point_2d)
            if self._is_valid_index(idx):
                self.grid[idx[1], idx[0]] += 1
        
        # Normalize
        if np.max(self.grid) > 0:
            self.grid = np.clip(self.grid / np.max(self.grid), 0, 1)
        
        return self.grid
    
    def _world_to_grid(self, point: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        idx = ((point - self.origin) / self.resolution).astype(int)
        return idx[0], idx[1]
    
    def _grid_to_world(self, idx: Tuple[int, int]) -> np.ndarray:
        """Convert grid indices to world coordinates."""
        point = self.origin + np.array(idx) * self.resolution
        return point
    
    def _is_valid_index(self, idx: Tuple[int, int]) -> bool:
        """Check if grid index is valid."""
        return 0 <= idx[0] < self.shape[1] and 0 <= idx[1] < self.shape[0]
    
    def apply_morphology(self, closing_size: int = 5, erosion_size: int = 1) -> np.ndarray:
        """
        Apply morphological operations to clean up grid.
        
        Args:
            closing_size: Size of closing operation (increased default for better wall continuity)
            erosion_size: Size of erosion operation
        
        Returns:
            Cleaned grid
        """
        if self.grid is None:
            return None
        
        # Binarize with lower threshold to preserve more wall points
        binary_grid = (self.grid > 0.05).astype(np.uint8)
        
        print(f"  Morphology input: {np.sum(binary_grid)} occupied cells")
        
        # Morphological closing (fill small gaps) - larger kernel for walls
        kernel_close = np.ones((closing_size, closing_size), np.uint8)
        closed = cv2.morphologyEx(binary_grid, cv2.MORPH_CLOSE, kernel_close)
        
        print(f"  After closing: {np.sum(closed)} occupied cells")
        
        # Light erosion to remove noise while preserving walls
        kernel_erode = np.ones((erosion_size, erosion_size), np.uint8)
        cleaned = cv2.erode(closed, kernel_erode, iterations=1)
        
        print(f"  After erosion: {np.sum(cleaned)} occupied cells")
        
        self.grid = cleaned.astype(np.float32)
        
        return self.grid
    
    def compute_distance_transform(self) -> np.ndarray:
        """
        Compute distance transform (distance to nearest occupied cell).
        
        Returns:
            Distance transform grid
        """
        if self.grid is None:
            return None
        
        binary = (self.grid > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        
        return dist
    
    def save_image(self, filename: str, colormap: str = 'gray'):
        """
        Save occupancy grid as image.
        
        Args:
            filename: Output filename
            colormap: Matplotlib colormap name
        """
        if self.grid is None:
            return
        
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap=colormap, origin='lower')
        plt.colorbar(label='Occupancy')
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        plt.title('Occupancy Grid')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
    
    def get_occupied_cells(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get coordinates of occupied cells.
        
        Args:
            threshold: Occupancy threshold
        
        Returns:
            Nx2 array of world coordinates
        """
        if self.grid is None:
            return np.array([])
        
        occupied_indices = np.argwhere(self.grid > threshold)
        
        # Convert to world coordinates
        world_coords = []
        for idx in occupied_indices:
            y, x = idx
            world_coord = self._grid_to_world((x, y))
            world_coords.append(world_coord)
        
        return np.array(world_coords)
    
    def compute_area(self) -> float:
        """
        Compute area of free space (unoccupied cells).
        
        Returns:
            Area in square meters
        """
        if self.grid is None:
            return 0.0
        
        # Count free cells
        free_cells = np.sum(self.grid < 0.5)
        
        # Convert to area
        cell_area = self.resolution ** 2
        area = free_cells * cell_area
        
        return area


def create_occupancy_grid_from_pointcloud(points: np.ndarray,
                                         resolution: float = 0.05,
                                         apply_cleanup: bool = True) -> Tuple[OccupancyGrid, np.ndarray]:
    """
    Create occupancy grid from point cloud.
    
    Args:
        points: Nx3 array of 3D points
        resolution: Grid resolution in meters
        apply_cleanup: Whether to apply morphological cleanup
    
    Returns:
        (OccupancyGrid object, grid array)
    """
    grid_obj = OccupancyGrid(resolution=resolution)
    grid = grid_obj.build_from_points(points)
    
    if apply_cleanup:
        grid = grid_obj.apply_morphology()
    
    return grid_obj, grid


def visualize_occupancy_grid(grid: np.ndarray,
                             resolution: float,
                             origin: np.ndarray,
                             filename: Optional[str] = None):
    """
    Visualize occupancy grid with proper scaling.
    
    Args:
        grid: Occupancy grid array
        resolution: Grid resolution
        origin: Grid origin (x_min, y_min)
        filename: Output filename (if None, display instead)
    """
    try:
        height, width = grid.shape
        extent = [
            origin[0],
            origin[0] + width * resolution,
            origin[1],
            origin[1] + height * resolution
        ]
        
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='binary', origin='lower', extent=extent)
        plt.colorbar(label='Occupancy')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('Occupancy Grid')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    except Exception as e:
        print(f"  Warning: Failed to visualize occupancy grid: {e}")
        if 'fig' in locals():
            plt.close(fig)

