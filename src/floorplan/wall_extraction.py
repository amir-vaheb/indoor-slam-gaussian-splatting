"""
Wall Extraction

Extract and vectorize walls from occupancy grid using Hough transform.
"""

import numpy as np
import cv2
from typing import List, Tuple
import geojson
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union


class Wall:
    """
    Represents a wall segment.
    
    Attributes:
        start: Start point [x, y] in world coordinates
        end: End point [x, y] in world coordinates
        length: Wall length in meters
    """
    
    def __init__(self, start: np.ndarray, end: np.ndarray):
        self.start = start.copy()
        self.end = end.copy()
        self.length = np.linalg.norm(end - start)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'start': self.start.tolist(),
            'end': self.end.tolist(),
            'length': float(self.length)
        }
    
    def to_geojson_feature(self) -> dict:
        """Convert to GeoJSON feature."""
        return {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': [
                    self.start.tolist(),
                    self.end.tolist()
                ]
            },
            'properties': {
                'length': float(self.length)
            }
        }


class WallExtractor:
    """
    Extract walls from occupancy grid using Hough transform.
    """
    
    def __init__(self,
                 hough_threshold: int = 50,
                 min_line_length: int = 30,
                 max_line_gap: int = 10,
                 min_wall_length: float = 0.3):
        """
        Initialize wall extractor.
        
        Args:
            hough_threshold: Hough transform threshold
            min_line_length: Minimum line length in pixels
            max_line_gap: Maximum gap between line segments
            min_wall_length: Minimum wall length in meters
        """
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_wall_length = min_wall_length
    
    def extract_walls(self,
                     occupancy_grid: np.ndarray,
                     resolution: float,
                     origin: np.ndarray) -> List[Wall]:
        """
        Extract walls from occupancy grid.
        
        Args:
            occupancy_grid: Binary occupancy grid (H, W)
            resolution: Grid resolution in meters
            origin: Grid origin (x_min, y_min)
        
        Returns:
            List of Wall objects
        """
        # Convert to binary image with lower threshold for better detection
        binary = (occupancy_grid > 0.3).astype(np.uint8) * 255
        
        # Debug output
        occupied_pixels = np.sum(binary > 0)
        print(f"  Binary grid: {binary.shape}, occupied pixels: {occupied_pixels}")
        
        # Apply dilation to make walls thicker before edge detection
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Edge detection with lower thresholds for indoor scenes
        edges = cv2.Canny(binary, 30, 100)
        edge_pixels = np.sum(edges > 0)
        print(f"  Edge pixels detected: {edge_pixels}")
        
        # Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            print(f"  WARNING: No lines detected by Hough transform!")
            print(f"  Hough params: threshold={self.hough_threshold}, minLen={self.min_line_length}, maxGap={self.max_line_gap}")
            return []
        
        print(f"  Hough detected {len(lines)} raw line segments")
        
        # Convert to Wall objects
        walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Convert from grid coordinates to world coordinates
            start_world = self._grid_to_world(np.array([x1, y1]), resolution, origin)
            end_world = self._grid_to_world(np.array([x2, y2]), resolution, origin)
            
            wall = Wall(start_world, end_world)
            
            # Filter short walls
            if wall.length >= self.min_wall_length:
                walls.append(wall)
        
        # Merge collinear walls
        walls = self._merge_collinear_walls(walls)
        
        return walls
    
    def _grid_to_world(self, grid_coord: np.ndarray, resolution: float, origin: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        # Grid coordinates are (x, y) in pixels
        # Note: y is flipped (origin='lower' in imshow)
        world = origin + grid_coord * resolution
        return world
    
    def _merge_collinear_walls(self, walls: List[Wall], angle_threshold: float = 0.1, distance_threshold: float = 0.1) -> List[Wall]:
        """
        Merge collinear wall segments.
        
        Args:
            walls: List of walls
            angle_threshold: Angle threshold for collinearity (radians)
            distance_threshold: Distance threshold for merging (meters)
        
        Returns:
            Merged walls
        """
        if len(walls) == 0:
            return []
        
        # Convert to Shapely LineStrings
        lines = [LineString([wall.start, wall.end]) for wall in walls]
        
        # Group collinear segments
        merged_groups = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            group = [i]
            used.add(i)
            
            for j, line2 in enumerate(lines):
                if j in used or j == i:
                    continue
                
                # Check if collinear and close
                if self._are_collinear(walls[i], walls[j], angle_threshold, distance_threshold):
                    group.append(j)
                    used.add(j)
            
            merged_groups.append(group)
        
        # Merge each group
        merged_walls = []
        for group in merged_groups:
            if len(group) == 1:
                merged_walls.append(walls[group[0]])
            else:
                # Merge walls in group
                merged_wall = self._merge_wall_group([walls[i] for i in group])
                if merged_wall is not None:
                    merged_walls.append(merged_wall)
        
        return merged_walls
    
    def _are_collinear(self, wall1: Wall, wall2: Wall, angle_threshold: float, distance_threshold: float) -> bool:
        """Check if two walls are collinear and close."""
        # Direction vectors
        dir1 = wall1.end - wall1.start
        dir1 = dir1 / (np.linalg.norm(dir1) + 1e-8)
        
        dir2 = wall2.end - wall2.start
        dir2 = dir2 / (np.linalg.norm(dir2) + 1e-8)
        
        # Check angle
        cos_angle = abs(np.dot(dir1, dir2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        if angle > angle_threshold:
            return False
        
        # Check distance between endpoints
        distances = [
            np.linalg.norm(wall1.start - wall2.start),
            np.linalg.norm(wall1.start - wall2.end),
            np.linalg.norm(wall1.end - wall2.start),
            np.linalg.norm(wall1.end - wall2.end)
        ]
        
        min_dist = min(distances)
        
        return min_dist < distance_threshold * 5  # More lenient distance threshold
    
    def _merge_wall_group(self, walls: List[Wall]) -> Wall:
        """Merge a group of walls into a single wall."""
        # Collect all endpoints
        points = []
        for wall in walls:
            points.append(wall.start)
            points.append(wall.end)
        
        points = np.array(points)
        
        # Fit a line to all points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # PCA to find principal direction
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Principal direction
        direction = eigenvectors[:, -1]
        
        # Project points onto line
        projections = np.dot(centered, direction)
        
        # Find extreme points
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        start = points[min_idx]
        end = points[max_idx]
        
        return Wall(start, end)


def compute_perimeter(walls: List[Wall]) -> float:
    """
    Compute total perimeter from walls.
    
    Args:
        walls: List of walls
    
    Returns:
        Total perimeter in meters
    """
    return sum(wall.length for wall in walls)


def save_walls_geojson(walls: List[Wall], filename: str):
    """
    Save walls to GeoJSON file.
    
    Args:
        walls: List of walls
        filename: Output filename
    """
    features = [wall.to_geojson_feature() for wall in walls]
    
    feature_collection = {
        'type': 'FeatureCollection',
        'features': features,
        'properties': {
            'num_walls': len(walls),
            'total_length': float(compute_perimeter(walls))
        }
    }
    
    with open(filename, 'w') as f:
        geojson.dump(feature_collection, f, indent=2)


def visualize_walls(walls: List[Wall],
                   occupancy_grid: np.ndarray,
                   resolution: float,
                   origin: np.ndarray,
                   filename: str):
    """
    Visualize walls on top of occupancy grid.
    
    Args:
        walls: List of walls
        occupancy_grid: Occupancy grid
        resolution: Grid resolution
        origin: Grid origin
        filename: Output filename
    """
    try:
        import matplotlib.pyplot as plt
        
        height, width = occupancy_grid.shape
        extent = [
            origin[0],
            origin[0] + width * resolution,
            origin[1],
            origin[1] + height * resolution
        ]
        
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(occupancy_grid, cmap='binary', origin='lower', extent=extent, alpha=0.5)
        
        # Draw walls
        for wall in walls:
            plt.plot(
                [wall.start[0], wall.end[0]],
                [wall.start[1], wall.end[1]],
                'r-', linewidth=2
            )
        
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title(f'Extracted Walls ({len(walls)} segments)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: Failed to visualize walls: {e}")
        if 'fig' in locals():
            plt.close(fig)

