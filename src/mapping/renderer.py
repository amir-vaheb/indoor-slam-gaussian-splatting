"""
Gaussian Splat Renderer

CPU-based renderer for Gaussian splats using alpha compositing.
"""

import numpy as np
from typing import List, Tuple
from src.mapping.gaussian_splats import GaussianSplat
import cv2


class GaussianSplatRenderer:
    """
    Simple CPU-based renderer for 3D Gaussian splats.
    
    Renders splats by:
    1. Projecting to 2D camera view
    2. Depth-sorting splats
    3. Rendering each as a 2D Gaussian
    4. Alpha-compositing (back-to-front)
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize renderer.
        
        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height
    
    def render(self,
               splats: List[GaussianSplat],
               pose: np.ndarray,
               intrinsics: np.ndarray) -> np.ndarray:
        """
        Render Gaussian splats from a given camera pose.
        
        Args:
            splats: List of Gaussian splats
            pose: Camera pose (4x4)
            intrinsics: Camera intrinsic matrix (3x3)
        
        Returns:
            Rendered RGB image (H, W, 3) in [0, 255]
        """
        # Initialize output image and depth buffer
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        alpha_buffer = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Transform splats to camera coordinates
        splat_data = []
        camera_to_world = pose
        world_to_camera = np.linalg.inv(camera_to_world)
        
        for splat in splats:
            # Transform splat center to camera coordinates
            point_world = np.append(splat.mean, 1.0)
            point_cam = world_to_camera @ point_world
            
            # Skip if behind camera
            if point_cam[2] <= 0:
                continue
            
            splat_data.append({
                'splat': splat,
                'pos_cam': point_cam[0:3],
                'depth': point_cam[2]
            })
        
        # Sort by depth (back to front for alpha compositing)
        splat_data.sort(key=lambda x: x['depth'], reverse=True)
        
        # Render each splat
        for data in splat_data:
            self._render_splat(
                data['splat'],
                data['pos_cam'],
                intrinsics,
                image,
                alpha_buffer
            )
        
        # Convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def _render_splat(self,
                     splat: GaussianSplat,
                     pos_cam: np.ndarray,
                     intrinsics: np.ndarray,
                     image: np.ndarray,
                     alpha_buffer: np.ndarray):
        """
        Render a single Gaussian splat.
        
        Args:
            splat: Gaussian splat
            pos_cam: Splat position in camera coordinates [x, y, z]
            intrinsics: Camera intrinsic matrix
            image: Output image buffer
            alpha_buffer: Alpha accumulation buffer
        """
        # Project to 2D
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_cam, y_cam, z_cam = pos_cam
        
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy
        
        # Skip if outside image
        if u < 0 or u >= self.width or v < 0 or v >= self.height:
            return
        
        # Estimate 2D scale (approximate projection of 3D scale)
        # Use average scale
        scale_3d = np.mean(splat.scale)
        scale_2d = fx * scale_3d / z_cam
        
        # Limit scale to reasonable range
        scale_2d = np.clip(scale_2d, 1.0, 50.0)
        
        # Define rendering region (3-sigma rule)
        radius = int(np.ceil(3 * scale_2d))
        
        u_min = max(0, int(u - radius))
        u_max = min(self.width, int(u + radius + 1))
        v_min = max(0, int(v - radius))
        v_max = min(self.height, int(v + radius + 1))
        
        # Render Gaussian in region
        for v_pix in range(v_min, v_max):
            for u_pix in range(u_min, u_max):
                # Distance from center
                du = u_pix - u
                dv = v_pix - v
                
                # Gaussian weight (isotropic for simplicity)
                dist_sq = (du**2 + dv**2) / (scale_2d**2)
                weight = np.exp(-0.5 * dist_sq)
                
                # Alpha value
                alpha = weight * splat.opacity
                
                # Skip if negligible contribution
                if alpha < 0.01:
                    continue
                
                # Alpha compositing (back-to-front)
                # Only composite if pixel is not fully opaque yet
                if alpha_buffer[v_pix, u_pix] < 0.99:
                    prev_alpha = alpha_buffer[v_pix, u_pix]
                    
                    # Blend color
                    image[v_pix, u_pix] = (
                        (1 - prev_alpha) * alpha * splat.rgb +
                        prev_alpha * image[v_pix, u_pix]
                    ) / ((1 - prev_alpha) * alpha + prev_alpha + 1e-8)
                    
                    # Update alpha buffer
                    alpha_buffer[v_pix, u_pix] = prev_alpha + (1 - prev_alpha) * alpha


def render_splat_preview(splats: List[GaussianSplat],
                         pose: np.ndarray,
                         intrinsics: np.ndarray,
                         width: int = 640,
                         height: int = 480) -> np.ndarray:
    """
    Convenience function to render a preview of splats.
    
    Args:
        splats: List of Gaussian splats
        pose: Camera pose (4x4)
        intrinsics: Camera intrinsic matrix (3x3)
        width: Image width
        height: Image height
    
    Returns:
        Rendered RGB image (H, W, 3)
    """
    renderer = GaussianSplatRenderer(width, height)
    return renderer.render(splats, pose, intrinsics)


def create_sample_view_pose(splats: List[GaussianSplat], 
                           distance_factor: float = 1.5) -> np.ndarray:
    """
    Create a sample camera pose for viewing splats.
    
    Args:
        splats: List of Gaussian splats
        distance_factor: Multiplier for viewing distance
    
    Returns:
        Camera pose (4x4)
    """
    if len(splats) == 0:
        return np.eye(4)
    
    # Compute center of splats
    centers = np.array([splat.mean for splat in splats])
    center = np.mean(centers, axis=0)
    
    # Compute bounding box size
    bbox_min = np.min(centers, axis=0)
    bbox_max = np.max(centers, axis=0)
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    
    # Camera position: offset from center
    distance = bbox_size * distance_factor
    camera_pos = center + np.array([0, 0, distance])
    
    # Look at center
    forward = center - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # Up vector
    up = np.array([0, -1, 0])
    
    # Right vector
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recompute up
    up = np.cross(right, forward)
    
    # Build rotation matrix (camera looks along -Z)
    R = np.eye(3)
    R[0, :] = right
    R[1, :] = up
    R[2, :] = -forward
    
    # Build pose
    pose = np.eye(4)
    pose[0:3, 0:3] = R
    pose[0:3, 3] = camera_pos
    
    return pose


def render_point_cloud_fallback(splats: List[GaussianSplat],
                                pose: np.ndarray,
                                intrinsics: np.ndarray,
                                width: int = 640,
                                height: int = 480) -> np.ndarray:
    """
    Fallback renderer that renders splats as simple points.
    Useful for debugging or when Gaussian rendering is too slow.
    
    Args:
        splats: List of Gaussian splats
        pose: Camera pose (4x4)
        intrinsics: Camera intrinsic matrix (3x3)
        width: Image width
        height: Image height
    
    Returns:
        Rendered RGB image (H, W, 3)
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    world_to_camera = np.linalg.inv(pose)
    
    for splat in splats:
        # Transform to camera coordinates
        point_world = np.append(splat.mean, 1.0)
        point_cam = world_to_camera @ point_world
        
        if point_cam[2] <= 0:
            continue
        
        # Project to 2D
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        u = int(fx * point_cam[0] / point_cam[2] + cx)
        v = int(fy * point_cam[1] / point_cam[2] + cy)
        
        if 0 <= u < width and 0 <= v < height:
            color = (splat.rgb * 255).astype(np.uint8)
            cv2.circle(image, (u, v), 2, color.tolist(), -1)
    
    return image

