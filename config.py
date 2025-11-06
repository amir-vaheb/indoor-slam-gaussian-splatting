"""
Configuration for Indoor SLAM & Floorplan System
"""

# Tracking Configuration
TRACKING_METHOD = "feature_pnp"  # Options: "feature_pnp", "icp"
FRAME_SKIP = 1  # Process every Nth frame (1 = all frames)
FEATURE_TYPE = "ORB"  # Options: "ORB", "AKAZE"
MAX_FEATURES = 2000
ICP_MAX_CORRESPONDENCE_DISTANCE = 0.05  # meters

# Pose Graph Optimization
SLIDING_WINDOW_SIZE = 20  # Number of recent poses to refine (increased from 10)
ENABLE_POSE_GRAPH = True

# Gaussian Splatting Configuration
VOXEL_SIZE = 0.05  # meters (5cm)
SPLAT_CLUSTER_RADIUS = 0.10  # meters
ISOTROPIC_PENALTY_THRESHOLD = 3.0  # max(scale)/min(scale) threshold
MIN_POINTS_PER_SPLAT = 5
OPACITY_DENSITY_SCALE = 0.8

# Floorplan Configuration
GRID_RESOLUTION = 0.05  # meters (5cm cells)
FLOOR_HEIGHT_MIN = 0.1  # meters above floor
FLOOR_HEIGHT_MAX = 2.0  # meters above floor
WALL_MIN_LENGTH = 0.2  # meters (reduced from 0.3 for shorter walls)
HOUGH_THRESHOLD = 20  # reduced from 50 for more sensitive detection
HOUGH_MIN_LINE_LENGTH = 15  # reduced from 30 pixels
HOUGH_MAX_LINE_GAP = 15  # increased from 10 to connect segments better

# Coverage Configuration
MIN_OBSERVATIONS_THRESHOLD = 2  # Reduced from 3 for better completeness metric
COVERAGE_GRID_RESOLUTION = 0.05  # meters (finer grid for better spatial resolution)

# Visualization
SHOW_INTERACTIVE_PLOTS = False  # Set to True to show interactive visualizations
RENDER_RESOLUTION = (640, 480)

# Data paths (relative to project root)
DATA_DIR = "data/rgbd_dataset_freiburg1_room"
RESULTS_DIR = "results"

