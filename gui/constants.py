"""GUI constants for PyroVision application."""

# Timer settings
TIMER_INTERVAL_MS = 33  # ~30 FPS for GUI updates

# FPS calculation
FPS_HISTORY_SIZE = 60  # Number of frames to track for FPS calculation

# Preview window sizes
PREVIEW_MIN_WIDTH = 320
PREVIEW_MIN_HEIGHT = 180

# Plot settings
PLOT_HISTORY_SIZE = 300  # Data points to show in rolling plots

# Video grid layout positions
VIDEO_GRID_RGB_ROW = 0
VIDEO_GRID_RGB_COL = 0
VIDEO_GRID_DET_ROW = 0
VIDEO_GRID_DET_COL = 1
VIDEO_GRID_IR_ROW = 1
VIDEO_GRID_IR_COL = 0
VIDEO_GRID_OVERLAY_ROW = 1
VIDEO_GRID_OVERLAY_COL = 1

# Status update interval
STATUS_UPDATE_INTERVAL_SEC = 0.1
