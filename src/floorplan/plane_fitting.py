"""
Plane Fitting

RANSAC-based plane fitting for floor and ceiling detection.
"""

import numpy as np
from typing import Optional, Tuple


class PlaneFitter:
    """
    RANSAC plane fitting for detecting horizontal planes (floor/ceiling).
    
    A plane is represented as: ax + by + cz + d = 0
    or in normal form: n·p + d = 0 where n = [a, b, c] is the normal vector
    """
    
    def __init__(self,
                 ransac_iterations: int = 1000,
                 inlier_threshold: float = 0.02,
                 min_inliers: int = 100):
        """
        Initialize plane fitter.
        
        Args:
            ransac_iterations: Number of RANSAC iterations
            inlier_threshold: Distance threshold for inliers (meters)
            min_inliers: Minimum number of inliers for valid plane
        """
        self.ransac_iterations = ransac_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers
    
    def fit_plane_ransac(self, points: np.ndarray) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        Fit plane to point cloud using RANSAC.
        
        Args:
            points: Nx3 array of points
        
        Returns:
            (normal, d, inliers) or None if fitting fails
            - normal: Unit normal vector [a, b, c]
            - d: Plane offset
            - inliers: Boolean mask of inlier points
        """
        if len(points) < 3:
            return None
        
        best_normal = None
        best_d = None
        best_inliers = None
        best_count = 0
        
        for _ in range(self.ransac_iterations):
            # Sample 3 random points
            indices = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[indices]
            
            # Compute plane from 3 points
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            # Skip if degenerate
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            
            normal = normal / norm
            d = -np.dot(normal, p1)
            
            # Count inliers
            distances = np.abs(np.dot(points, normal) + d)
            inliers = distances < self.inlier_threshold
            count = np.sum(inliers)
            
            if count > best_count:
                best_count = count
                best_normal = normal
                best_d = d
                best_inliers = inliers
        
        if best_count < self.min_inliers:
            return None
        
        # Refine plane with all inliers
        inlier_points = points[best_inliers]
        normal, d = self._fit_plane_least_squares(inlier_points)
        
        return normal, d, best_inliers
    
    def _fit_plane_least_squares(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit plane to points using least squares.
        
        Args:
            points: Nx3 array of points
        
        Returns:
            (normal, d)
        """
        # Center points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # SVD
        _, _, Vt = np.linalg.svd(centered)
        
        # Normal is last singular vector
        normal = Vt[2, :]
        
        # Compute d
        d = -np.dot(normal, centroid)
        
        return normal, d
    
    def is_horizontal(self, normal: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Check if plane is horizontal (normal close to vertical).
        
        Args:
            normal: Plane normal vector
            threshold: Threshold for deviation from vertical (radians)
        
        Returns:
            True if plane is horizontal
        """
        # Vertical direction (assuming Z is up)
        vertical = np.array([0, 0, 1])
        
        # Angle between normal and vertical
        cos_angle = np.abs(np.dot(normal, vertical))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return angle < threshold


def detect_floor_ceiling(points: np.ndarray,
                        remove_detected: bool = True) -> Tuple[Optional[Tuple], Optional[Tuple], np.ndarray]:
    """
    Detect floor and ceiling planes from point cloud.
    
    Args:
        points: Nx3 array of points
        remove_detected: If True, return points with floor/ceiling removed
    
    Returns:
        (floor_plane, ceiling_plane, remaining_points)
        - floor_plane: (normal, d, inliers) or None
        - ceiling_plane: (normal, d, inliers) or None
        - remaining_points: Points not belonging to floor or ceiling
    """
    fitter = PlaneFitter()
    
    floor_plane = None
    ceiling_plane = None
    remaining_mask = np.ones(len(points), dtype=bool)
    
    # Find floor (lowest horizontal plane)
    floor_result = fitter.fit_plane_ransac(points)
    
    if floor_result is not None:
        normal, d, inliers = floor_result
        
        if fitter.is_horizontal(normal):
            # Ensure normal points up
            if normal[2] < 0:
                normal = -normal
                d = -d
            
            floor_plane = (normal, d, inliers)
            remaining_mask &= ~inliers
    
    # Find ceiling (highest horizontal plane among remaining points)
    if np.sum(remaining_mask) > 100:
        remaining_points = points[remaining_mask]
        ceiling_result = fitter.fit_plane_ransac(remaining_points)
        
        if ceiling_result is not None:
            normal, d, ceiling_inliers_local = ceiling_result
            
            if fitter.is_horizontal(normal):
                # Ensure normal points down
                if normal[2] > 0:
                    normal = -normal
                    d = -d
                
                # Map inliers back to original indices
                ceiling_inliers = np.zeros(len(points), dtype=bool)
                ceiling_inliers[remaining_mask] = ceiling_inliers_local
                
                ceiling_plane = (normal, d, ceiling_inliers)
                remaining_mask &= ~ceiling_inliers
    
    remaining_points = points[remaining_mask]
    
    return floor_plane, ceiling_plane, remaining_points


def get_floor_height(floor_plane: Optional[Tuple]) -> float:
    """
    Get floor height (Z coordinate).
    
    Args:
        floor_plane: (normal, d, inliers)
    
    Returns:
        Floor height or 0.0 if not available
    """
    if floor_plane is None:
        return 0.0
    
    normal, d, _ = floor_plane
    
    # Floor plane equation: n·p + d = 0
    # For a horizontal plane with normal [0, 0, 1]: z + d = 0 → z = -d
    # But normal might not be exactly [0, 0, 1]
    
    # Find z when x=0, y=0
    if abs(normal[2]) > 0.01:
        z = -d / normal[2]
    else:
        z = 0.0
    
    return z


def filter_points_by_height(points: np.ndarray,
                            floor_height: float,
                            min_height: float = 0.1,
                            max_height: float = 2.0) -> np.ndarray:
    """
    Filter points by height above floor.
    
    Args:
        points: Nx3 array of points
        floor_height: Floor Z coordinate
        min_height: Minimum height above floor
        max_height: Maximum height above floor
    
    Returns:
        Filtered points
    """
    heights = points[:, 2] - floor_height
    mask = (heights >= min_height) & (heights <= max_height)
    return points[mask]

