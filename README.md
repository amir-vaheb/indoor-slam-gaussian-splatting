# Indoor SLAM & Floorplan with Gaussian Splatting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

A complete RGB-D SLAM system that produces camera trajectories, 3D Gaussian splat maps, 2D floorplans, and coverage analysis from indoor sequences.

## Features

- **Dual Tracking Methods**: Feature-based PnP+RANSAC or RGB-D ICP
- **3D Gaussian Splatting**: Lightweight anisotropic Gaussians with CPU rendering
- **2D Floorplan Extraction**: Automatic wall detection and vectorization
- **Coverage Analysis**: Observation-based completeness heatmaps
- **Visualization Tools**: Interactive 3D viewers, trajectory plots, and heatmaps

## Installation

```bash
# Clone repository
git clone <repo-url>
cd indoor-slam-gaussian-splatting

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy, OpenCV, SciPy, Open3D, Matplotlib
- 2GB disk space (for dataset)
- 4GB RAM minimum

## Data Setup

This system expects data in TUM RGB-D format. Download the test dataset:

```bash
cd data
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
tar -xzf rgbd_dataset_freiburg1_room.tgz
```

## Usage

### Basic Usage

```bash
python main.py --data data/rgbd_dataset_freiburg1_room
```

### Configuration Options

Edit `config.py` to change:
- `TRACKING_METHOD`: Choose "feature_pnp" or "icp"
- `GRID_RESOLUTION`: Floorplan grid cell size
- `MIN_OBSERVATIONS_THRESHOLD`: Coverage completeness threshold
- `SHOW_INTERACTIVE_PLOTS`: Enable interactive visualizations

### Command Line Options

```bash
python main.py --data <path> --tracking <method> --visualize
```

## Output Files

All outputs are saved to `results/`:

- `trajectory.txt`: Estimated camera poses (TUM format)
- `splats.json`: 3D Gaussian splats
- `splat_preview.png`: Rendered view from sample pose
- `floorplan.png`: 2D occupancy grid
- `walls.geojson`: Vectorized wall segments
- `coverage_heatmap.png`: Observation count heatmap
- `metrics.json`: Summary statistics

## Architecture

### Project Structure

```
.
├── config.py                  # Configuration parameters
├── main.py                    # Main pipeline
├── src/
│   ├── tracking/              # Camera pose estimation
│   │   ├── feature_pnp.py    # Feature-based tracking
│   │   ├── icp.py            # ICP tracking
│   │   └── pose_graph.py     # Drift reduction
│   ├── mapping/               # 3D reconstruction
│   │   ├── gaussian_splats.py # Splat creation
│   │   └── renderer.py        # CPU renderer
│   ├── floorplan/             # 2D floorplan
│   │   ├── plane_fitting.py   # Floor detection
│   │   ├── occupancy_grid.py  # Grid generation
│   │   └── wall_extraction.py # Wall vectorization
│   ├── coverage/              # Coverage analysis
│   │   └── coverage_map.py    # Observation tracking
│   └── utils/                 # Utilities
│       ├── data_loader.py     # TUM format parser
│       └── visualization.py   # Plotting tools
└── results/                   # Output directory
```

## Algorithm Overview

### 1. Tracking

**Feature PnP Method:**
- Extract ORB/AKAZE features from RGB frames
- Match features between consecutive frames using BFMatcher
- Back-project 2D keypoints to 3D using depth
- Estimate relative pose with PnP+RANSAC
- Optional sliding-window pose graph optimization

**RGB-D ICP Method:**
- Convert RGB-D frames to colored point clouds
- Align using Open3D's colored ICP or point-to-plane ICP
- Use previous pose as initialization
- Optional pose graph refinement

### 2. Gaussian Splatting

- Accumulate 3D points from all frames using estimated poses
- Voxel downsample for efficiency (5cm voxels)
- Cluster nearby points using radius search
- For each cluster:
  - **Mean**: Centroid of cluster points
  - **Scale**: sqrt(eigenvalues) from covariance matrix
  - **RGB**: Average color of cluster
  - **Opacity**: Based on local point density
- Apply isotropic regularizer to prevent extreme elongation

**Rendering:**
- Project Gaussians to camera view
- Depth-sort splats
- Alpha-composite (back-to-front or front-to-back)

### 3. Floorplan Extraction

- **Floor Detection**: RANSAC plane fitting on accumulated point cloud
- **Height Filtering**: Keep points 0.1m-2.0m above floor (walls)
- **Occupancy Grid**: Project vertical structures to 2D grid (5cm cells)
- **Cleanup**: Morphological operations (closing, erosion)
- **Wall Vectorization**: Hough line transform + merge collinear segments

### 4. Coverage Analysis

- Maintain 2D grid aligned with floorplan
- Count observations per cell as frames are processed
- Completeness = % of floor cells with ≥ N observations (N=3)
- Identify and report low-coverage regions

## Key Design Decisions

- **Tracking**: Both methods supported; Feature PnP is faster, ICP more robust
- **Drift**: Sliding-window pose graph (no loop closure)
- **Splats**: Covariance-based anisotropic with isotropic penalty
- **Floorplan**: Height-band filtering + Hough transform
- **Coverage**: Ray/point-based observation counting

## Evaluation

If ground truth poses are available, the system computes:
- **ATE RMSE**: Absolute Trajectory Error (meters)
- **Trajectory length**: Total path distance (meters)

Additional metrics:
- Floor area (m²)
- Perimeter (m)
- Coverage completeness (%)
- Number of Gaussian splats

## Assumptions & Limitations

- Small indoor scenes (~20m²)
- Handheld camera motion (not too fast)
- Moderate drift acceptable (no loop closure)
- CPU-only rendering (sufficient for preview)
- Offline processing (not real-time)
- Manhattan-world assumption for floorplan extraction
- Static scenes (no dynamic objects)
