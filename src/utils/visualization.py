"""
Visualization Utilities

Tools for visualizing trajectories, point clouds, and other data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import List, Optional
from src.mapping.gaussian_splats import GaussianSplat


def plot_trajectory_2d(poses: np.ndarray,
                      gt_poses: Optional[np.ndarray] = None,
                      filename: Optional[str] = None):
    """
    Plot camera trajectory in 2D (top-down view).
    
    Args:
        poses: Nx4x4 estimated poses
        gt_poses: Nx4x4 ground truth poses (optional)
        filename: Save to file if provided
    """
    plt.figure(figsize=(10, 10))
    
    # Extract positions
    positions = poses[:, 0:3, 3]
    
    # Plot estimated trajectory
    plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Estimated')
    plt.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start')
    plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='s', label='End')
    
    # Plot ground truth if available
    if gt_poses is not None:
        gt_positions = gt_poses[:, 0:3, 3]
        plt.plot(gt_positions[:, 0], gt_positions[:, 1], 'r--', linewidth=2, label='Ground Truth', alpha=0.7)
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Camera Trajectory (Top View)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_trajectory_3d(poses: np.ndarray,
                      gt_poses: Optional[np.ndarray] = None,
                      filename: Optional[str] = None):
    """
    Plot camera trajectory in 3D.
    
    Args:
        poses: Nx4x4 estimated poses
        gt_poses: Nx4x4 ground truth poses (optional)
        filename: Save to file if provided
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = poses[:, 0:3, 3]
    
    # Plot estimated trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Estimated')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, marker='s', label='End')
    
    # Plot ground truth if available
    if gt_poses is not None:
        gt_positions = gt_poses[:, 0:3, 3]
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'r--', linewidth=2, label='Ground Truth', alpha=0.7)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Camera Trajectory (3D)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_trajectory_comparison(est_poses: np.ndarray,
                              gt_poses: np.ndarray,
                              filename: Optional[str] = None):
    """
    Side-by-side comparison of estimated vs ground truth trajectory.
    
    Args:
        est_poses: Nx4x4 estimated poses
        gt_poses: Mx4x4 ground truth poses
        filename: Save to file if provided
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract positions
    est_positions = est_poses[:, 0:3, 3]
    gt_positions = gt_poses[:, 0:3, 3]
    
    # Plot XY
    axes[0].plot(est_positions[:, 0], est_positions[:, 1], 'b-', linewidth=2, label='Estimated')
    axes[0].plot(gt_positions[:, 0], gt_positions[:, 1], 'r--', linewidth=2, label='Ground Truth')
    axes[0].scatter(est_positions[0, 0], est_positions[0, 1], c='green', s=100, marker='o')
    axes[0].set_xlabel('X (meters)')
    axes[0].set_ylabel('Y (meters)')
    axes[0].set_title('Top View (XY)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Plot XZ
    axes[1].plot(est_positions[:, 0], est_positions[:, 2], 'b-', linewidth=2, label='Estimated')
    axes[1].plot(gt_positions[:, 0], gt_positions[:, 2], 'r--', linewidth=2, label='Ground Truth')
    axes[1].scatter(est_positions[0, 0], est_positions[0, 2], c='green', s=100, marker='o')
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Z (meters)')
    axes[1].set_title('Side View (XZ)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_ate_over_time(est_poses: np.ndarray,
                      est_timestamps: List[float],
                      gt_poses: np.ndarray,
                      gt_timestamps: List[float],
                      filename: Optional[str] = None):
    """
    Plot absolute trajectory error over time.
    
    Args:
        est_poses: Estimated poses
        est_timestamps: Estimated timestamps
        gt_poses: Ground truth poses
        gt_timestamps: Ground truth timestamps
        filename: Save to file if provided
    """
    errors = []
    times = []
    
    for i, ts in enumerate(est_timestamps):
        # Find closest ground truth
        idx = np.argmin(np.abs(np.array(gt_timestamps) - ts))
        dt = abs(gt_timestamps[idx] - ts)
        
        if dt > 0.05:
            continue
        
        # Translation error
        est_trans = est_poses[i][0:3, 3]
        gt_trans = gt_poses[idx][0:3, 3]
        error = np.linalg.norm(est_trans - gt_trans)
        
        errors.append(error)
        times.append(ts)
    
    if len(errors) == 0:
        print("No matching timestamps for ATE plot")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, errors, 'b-', linewidth=2)
    plt.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}m')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Absolute Translation Error (meters)')
    plt.title('Absolute Trajectory Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_point_cloud(points: np.ndarray,
                         colors: Optional[np.ndarray] = None,
                         window_name: str = "Point Cloud"):
    """
    Visualize point cloud using Open3D.
    
    Args:
        points: Nx3 array of points
        colors: Nx3 array of colors [0, 1] (optional)
        window_name: Window title
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals for better visualization
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def visualize_splats_as_pointcloud(splats: List[GaussianSplat],
                                   window_name: str = "Gaussian Splats"):
    """
    Visualize Gaussian splats as colored point cloud.
    
    Args:
        splats: List of Gaussian splats
        window_name: Window title
    """
    if len(splats) == 0:
        print("No splats to visualize")
        return
    
    points = np.array([splat.mean for splat in splats])
    colors = np.array([splat.rgb for splat in splats])
    
    visualize_point_cloud(points, colors, window_name)


def visualize_trajectory_with_pointcloud(poses: np.ndarray,
                                        points: np.ndarray,
                                        colors: Optional[np.ndarray] = None):
    """
    Visualize camera trajectory together with point cloud.
    
    Args:
        poses: Nx4x4 camera poses
        points: Mx3 point cloud
        colors: Mx3 colors (optional)
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create camera trajectory
    camera_positions = poses[:, 0:3, 3]
    
    # Create line set for trajectory
    lines = [[i, i+1] for i in range(len(camera_positions)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(camera_positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red
    
    # Create camera frustums (simplified as small frames)
    frames = []
    for i in range(0, len(poses), max(1, len(poses)//10)):  # Show every 10th camera
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        frame.transform(poses[i])
        frames.append(frame)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, line_set] + frames, window_name="Trajectory + Point Cloud")


def create_summary_figure(est_poses: np.ndarray,
                         gt_poses: Optional[np.ndarray],
                         coverage_map,
                         metrics: dict,
                         filename: str):
    """
    Create comprehensive summary figure with multiple subplots.
    
    Args:
        est_poses: Estimated poses
        gt_poses: Ground truth poses (optional)
        coverage_map: CoverageMap object
        metrics: Metrics dictionary
        filename: Output filename
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Trajectory top view
    ax1 = plt.subplot(2, 2, 1)
    positions = est_poses[:, 0:3, 3]
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Estimated')
    if gt_poses is not None:
        gt_positions = gt_poses[:, 0:3, 3]
        ax1.plot(gt_positions[:, 0], gt_positions[:, 1], 'r--', linewidth=2, label='GT', alpha=0.7)
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Camera Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Coverage heatmap
    ax2 = plt.subplot(2, 2, 2)
    if coverage_map.grid is not None:
        height, width = coverage_map.shape
        extent = [
            coverage_map.origin[0],
            coverage_map.origin[0] + width * coverage_map.resolution,
            coverage_map.origin[1],
            coverage_map.origin[1] + height * coverage_map.resolution
        ]
        im = ax2.imshow(coverage_map.grid, cmap='YlOrRd', origin='lower', extent=extent)
        plt.colorbar(im, ax=ax2, label='Observations')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Coverage ({coverage_map.compute_completeness():.1%})')
    
    # 3. Metrics text
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    metrics_text = "System Metrics:\n\n"
    if 'ate_rmse' in metrics and metrics['ate_rmse'] is not None:
        metrics_text += f"ATE RMSE: {metrics['ate_rmse']:.3f} m\n"
    if 'trajectory_length' in metrics:
        metrics_text += f"Trajectory Length: {metrics['trajectory_length']:.2f} m\n"
    if 'num_frames' in metrics:
        metrics_text += f"Number of Frames: {metrics['num_frames']}\n"
    if 'floor_area_m2' in metrics:
        metrics_text += f"Floor Area: {metrics['floor_area_m2']:.2f} m²\n"
    if 'perimeter_m' in metrics:
        metrics_text += f"Perimeter: {metrics['perimeter_m']:.2f} m\n"
    if 'coverage_completeness' in metrics:
        metrics_text += f"Coverage: {metrics['coverage_completeness']:.1%}\n"
    if 'num_splats' in metrics:
        metrics_text += f"Gaussian Splats: {metrics['num_splats']}\n"
    
    ax3.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center', family='monospace')
    ax3.set_title('Performance Metrics')
    
    # 4. Height profile
    ax4 = plt.subplot(2, 2, 4)
    heights = positions[:, 2]
    ax4.plot(heights, 'g-', linewidth=2)
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Z Height (m)')
    ax4.set_title('Camera Height Profile')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def print_metrics_summary(metrics: dict):
    """
    Print metrics summary to console.
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "="*60)
    print("SLAM SYSTEM METRICS")
    print("="*60)
    
    if 'ate_rmse' in metrics and metrics['ate_rmse'] is not None:
        print(f"  ATE RMSE:              {metrics['ate_rmse']:.4f} m")
    
    if 'trajectory_length' in metrics:
        print(f"  Trajectory Length:     {metrics['trajectory_length']:.2f} m")
    
    if 'num_frames' in metrics:
        print(f"  Frames Processed:      {metrics['num_frames']}")
    
    if 'floor_area_m2' in metrics:
        print(f"  Floor Area:            {metrics['floor_area_m2']:.2f} m²")
    
    if 'perimeter_m' in metrics:
        print(f"  Perimeter:             {metrics['perimeter_m']:.2f} m")
    
    if 'coverage_completeness' in metrics:
        print(f"  Coverage Completeness: {metrics['coverage_completeness']:.1%}")
    
    if 'num_splats' in metrics:
        print(f"  Gaussian Splats:       {metrics['num_splats']}")
    
    print("="*60 + "\n")

