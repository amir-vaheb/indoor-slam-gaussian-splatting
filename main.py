#!/usr/bin/env python3
"""
Indoor SLAM & Floorplan System

Main pipeline integrating tracking, mapping, floorplan, and coverage.
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Set matplotlib backend to non-interactive BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent crashes

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.utils.data_loader import TUMDataLoader, save_trajectory_tum
from src.tracking.feature_pnp import FeaturePnPTracker
from src.tracking.icp import ICPTracker
from src.tracking.pose_graph import PoseGraphOptimizer
from src.mapping.gaussian_splats import GaussianSplatMapper, save_splats
from src.mapping.renderer import render_splat_preview, create_sample_view_pose
from src.floorplan.plane_fitting import detect_floor_ceiling, get_floor_height, filter_points_by_height
from src.floorplan.occupancy_grid import create_occupancy_grid_from_pointcloud, visualize_occupancy_grid
from src.floorplan.wall_extraction import WallExtractor, save_walls_geojson, compute_perimeter, visualize_walls
from src.coverage.coverage_map import CoverageMap, compute_trajectory_metrics, save_metrics_json
from src.utils.visualization import (
    plot_trajectory_2d, plot_trajectory_3d, plot_trajectory_comparison,
    print_metrics_summary, create_summary_figure
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Indoor SLAM & Floorplan System')
    parser.add_argument('--data', type=str, default=config.DATA_DIR,
                       help='Path to TUM dataset directory')
    parser.add_argument('--tracking', type=str, default=config.TRACKING_METHOD,
                       choices=['feature_pnp', 'icp'],
                       help='Tracking method')
    parser.add_argument('--visualize', action='store_true',
                       help='Show interactive visualizations')
    parser.add_argument('--output', type=str, default=config.RESULTS_DIR,
                       help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("INDOOR SLAM & FLOORPLAN SYSTEM")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Tracking: {args.tracking}")
    print(f"Output: {args.output}")
    print("="*60 + "\n")
    
    # ========== 1. Load Data ==========
    print("[1/7] Loading data...")
    loader = TUMDataLoader(args.data, frame_skip=config.FRAME_SKIP)
    intrinsics = loader.get_intrinsic_matrix()
    
    # ========== 2. Tracking ==========
    print(f"[2/7] Running tracking ({args.tracking})...")
    
    # Initialize tracker
    if args.tracking == 'feature_pnp':
        tracker = FeaturePnPTracker(
            feature_type=config.FEATURE_TYPE,
            max_features=config.MAX_FEATURES
        )
    else:  # icp
        tracker = ICPTracker(
            max_correspondence_distance=config.ICP_MAX_CORRESPONDENCE_DISTANCE
        )
    
    # Initialize pose graph optimizer
    pose_graph = PoseGraphOptimizer(window_size=config.SLIDING_WINDOW_SIZE) if config.ENABLE_POSE_GRAPH else None
    
    # Track all frames
    poses = []
    timestamps = []
    prev_pose = np.eye(4)
    
    for i in tqdm(range(len(loader)), desc="Tracking"):
        frame = loader[i]
        
        success, pose, info = tracker.track(
            frame['rgb'],
            frame['depth'],
            intrinsics,
            prev_pose
        )
        
        if success:
            poses.append(pose.copy())
            timestamps.append(frame['timestamp'])
            
            # Add to pose graph
            if pose_graph is not None:
                if len(poses) == 1:
                    pose_graph.add_pose(pose, frame['timestamp'])
                else:
                    rel_transform = np.linalg.inv(prev_pose) @ pose
                    pose_graph.add_pose(pose, frame['timestamp'], rel_transform)
                
                # Optimize periodically
                if len(poses) % config.SLIDING_WINDOW_SIZE == 0:
                    optimized_poses = pose_graph.optimize()
                    poses = optimized_poses.copy()
            
            prev_pose = pose.copy()
        else:
            # Keep previous pose on failure
            poses.append(prev_pose.copy())
            timestamps.append(frame['timestamp'])
    
    poses = np.array(poses)
    print(f"  Tracked {len(poses)} frames")
    
    # Save trajectory
    trajectory_file = output_dir / 'trajectory.txt'
    save_trajectory_tum(trajectory_file, timestamps, poses)
    print(f"  Saved trajectory to {trajectory_file}")
    
    # ========== 3. Gaussian Splatting ==========
    print("[3/7] Building Gaussian splat map...")
    
    splat_mapper = GaussianSplatMapper(
        voxel_size=config.VOXEL_SIZE,
        cluster_radius=config.SPLAT_CLUSTER_RADIUS,
        min_points_per_splat=config.MIN_POINTS_PER_SPLAT,
        isotropic_penalty_threshold=config.ISOTROPIC_PENALTY_THRESHOLD
    )
    
    # Add frames to mapper (subsample for efficiency)
    frame_step = max(1, len(poses) // 100)
    for i in tqdm(range(0, len(loader), frame_step), desc="Mapping"):
        if i >= len(poses):
            break
        frame = loader[i]
        splat_mapper.add_frame(frame['rgb'], frame['depth'], poses[i], intrinsics)
    
    # Build splats
    splats = splat_mapper.build_splats()
    print(f"  Created {len(splats)} splats")
    
    # Save splats
    splats_file = output_dir / 'splats.json'
    save_splats(splats, splats_file)
    print(f"  Saved splats to {splats_file}")
    
    # Render preview
    if len(splats) > 0:
        sample_pose = create_sample_view_pose(splats, distance_factor=2.0)
        preview_image = render_splat_preview(splats, sample_pose, intrinsics)
        
        import cv2
        preview_file = output_dir / 'splat_preview.png'
        cv2.imwrite(str(preview_file), cv2.cvtColor(preview_image, cv2.COLOR_RGB2BGR))
        print(f"  Saved preview to {preview_file}")
    
    # ========== 4. Floorplan Extraction ==========
    print("[4/7] Extracting floorplan...")
    
    # Accumulate point cloud
    all_points = []
    for i in tqdm(range(0, len(loader), frame_step), desc="Accumulating points"):
        if i >= len(poses):
            break
        frame = loader[i]
        
        # Back-project to 3D
        height, width = frame['depth'].shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        for v in range(0, height, 5):
            for u in range(0, width, 5):
                z = frame['depth'][v, u]
                if z <= 0 or z > 10.0:
                    continue
                
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                point_cam = np.array([x, y, z, 1.0])
                point_world = poses[i] @ point_cam
                all_points.append(point_world[0:3])
    
    all_points = np.array(all_points)
    print(f"  Accumulated {len(all_points)} points")
    
    # Detect floor and ceiling with error handling
    try:
        floor_plane, ceiling_plane, wall_points = detect_floor_ceiling(all_points)
        floor_height = get_floor_height(floor_plane)
        print(f"  Floor height: {floor_height:.2f} m")
        
        # Filter by height
        wall_points = filter_points_by_height(
            wall_points if len(wall_points) > 0 else all_points,
            floor_height,
            config.FLOOR_HEIGHT_MIN,
            config.FLOOR_HEIGHT_MAX
        )
        print(f"  Filtered to {len(wall_points)} wall points")
    except Exception as e:
        print(f"  Warning: Floor detection failed ({e}), using all points for walls")
        wall_points = all_points
        floor_height = 0.0
    
    # Create occupancy grid with error handling
    try:
        grid_obj, grid = create_occupancy_grid_from_pointcloud(
            wall_points,
            resolution=config.GRID_RESOLUTION,
            apply_cleanup=True
        )
        
        # Save floorplan image
        floorplan_file = output_dir / 'floorplan.png'
        visualize_occupancy_grid(grid, grid_obj.resolution, grid_obj.origin, floorplan_file)
        print(f"  Saved floorplan to {floorplan_file}")
        
        # Extract walls with error handling
        try:
            wall_extractor = WallExtractor(
                hough_threshold=config.HOUGH_THRESHOLD,
                min_line_length=config.HOUGH_MIN_LINE_LENGTH,
                max_line_gap=config.HOUGH_MAX_LINE_GAP,
                min_wall_length=config.WALL_MIN_LENGTH
            )
            
            walls = wall_extractor.extract_walls(grid, grid_obj.resolution, grid_obj.origin)
            print(f"  Extracted {len(walls)} walls")
            
            # Save walls
            walls_file = output_dir / 'walls.geojson'
            save_walls_geojson(walls, walls_file)
            print(f"  Saved walls to {walls_file}")
            
            # Compute floorplan metrics
            floor_area = grid_obj.compute_area()
            perimeter = compute_perimeter(walls)
            print(f"  Floor area: {floor_area:.2f} m²")
            print(f"  Perimeter: {perimeter:.2f} m")
        except Exception as e:
            print(f"  Error during wall extraction: {e}")
            import traceback
            traceback.print_exc()
            # Create empty walls file
            walls = []
            walls_file = output_dir / 'walls.geojson'
            save_walls_geojson(walls, walls_file)
            floor_area = grid_obj.compute_area() if 'grid_obj' in locals() else 0.0
            perimeter = 0.0
            print(f"  Floor area: {floor_area:.2f} m²")
            print(f"  Perimeter: {perimeter:.2f} m")
    except Exception as e:
        print(f"  Error during occupancy grid creation: {e}")
        import traceback
        traceback.print_exc()
        walls = []
        floor_area = 0.0
        perimeter = 0.0
    
    # ========== 5. Coverage Analysis ==========
    print("[5/7] Computing coverage...")
    
    # Initialize coverage map
    min_xy = np.min(all_points[:, 0:2], axis=0)
    max_xy = np.max(all_points[:, 0:2], axis=0)
    
    coverage_map = CoverageMap(
        resolution=config.COVERAGE_GRID_RESOLUTION,
        min_observations=config.MIN_OBSERVATIONS_THRESHOLD
    )
    coverage_map.initialize_from_bounds(min_xy, max_xy)
    
    # Add observations
    for i in tqdm(range(0, len(loader), frame_step), desc="Coverage"):
        if i >= len(poses):
            break
        frame = loader[i]
        coverage_map.add_observation(
            frame['rgb'],
            frame['depth'],
            poses[i],
            intrinsics,
            floor_height
        )
    
    completeness = coverage_map.compute_completeness()
    print(f"  Coverage completeness: {completeness:.1%}")
    
    # Save coverage heatmap
    heatmap_file = output_dir / 'coverage_heatmap.png'
    coverage_map.save_heatmap(heatmap_file)
    print(f"  Saved heatmap to {heatmap_file}")
    
    # ========== 6. Compute Metrics ==========
    print("[6/7] Computing metrics...")
    
    # Get ground truth if available
    gt_poses = None
    gt_timestamps = None
    if loader.has_groundtruth:
        gt_poses = loader.get_all_groundtruth_poses()
        gt_timestamps = loader.gt_timestamps
    
    # Trajectory metrics
    traj_metrics = compute_trajectory_metrics(poses, timestamps, gt_poses, gt_timestamps)
    
    # Combined metrics
    metrics = {
        **traj_metrics,
        'floor_area_m2': float(floor_area),
        'perimeter_m': float(perimeter),
        'coverage_completeness': float(completeness),
        'num_splats': len(splats),
        'num_walls': len(walls),
        **coverage_map.get_statistics()
    }
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    save_metrics_json(metrics, metrics_file)
    print(f"  Saved metrics to {metrics_file}")
    
    # Print summary
    print_metrics_summary(metrics)
    
    # ========== 7. Visualization ==========
    print("[7/7] Generating visualizations...")
    
    # Trajectory plots
    traj_2d_file = output_dir / 'trajectory_2d.png'
    plot_trajectory_2d(poses, gt_poses, traj_2d_file)
    
    traj_3d_file = output_dir / 'trajectory_3d.png'
    plot_trajectory_3d(poses, gt_poses, traj_3d_file)
    
    if gt_poses is not None:
        traj_comp_file = output_dir / 'trajectory_comparison.png'
        plot_trajectory_comparison(poses, gt_poses, traj_comp_file)
    
    # Summary figure
    summary_file = output_dir / 'summary.png'
    create_summary_figure(poses, gt_poses, coverage_map, metrics, summary_file)
    
    # Wall visualization
    walls_viz_file = output_dir / 'walls_visualization.png'
    visualize_walls(walls, grid, grid_obj.resolution, grid_obj.origin, walls_viz_file)
    
    print(f"  Saved visualizations to {output_dir}")
    
    # Interactive visualizations
    if args.visualize:
        print("\nLaunching interactive visualizations...")
        from src.utils.visualization import visualize_splats_as_pointcloud
        visualize_splats_as_pointcloud(splats)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nOutput files:")
    print("  - trajectory.txt          : Camera poses (TUM format)")
    print("  - splats.json             : Gaussian splats")
    print("  - splat_preview.png       : Rendered splat view")
    print("  - floorplan.png           : Occupancy grid")
    print("  - walls.geojson           : Extracted wall segments")
    print("  - coverage_heatmap.png    : Coverage analysis")
    print("  - metrics.json            : Performance metrics")
    print("  - trajectory_*.png        : Trajectory visualizations")
    print("  - summary.png             : Summary figure")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

