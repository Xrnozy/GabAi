# ---------------------------------------------------------------------------
# config.py – All constants, model paths, shared state, and model loading
# ---------------------------------------------------------------------------
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
BASE_DIR              = Path(__file__).resolve().parent
DEPTH_REPO_DIR        = BASE_DIR / "depth_anything_v2"
DEPTH_CHECKPOINT_PATH = BASE_DIR / "models" / "depth" / "depth_anything_v2.pth"
POTHOLE_MODEL_PATH    = BASE_DIR / "models" / "segmentation" / "Yolov8-fintuned-on-potholes.pt"
ROAD_PATH_MODEL_PATH  = BASE_DIR / "models" / "segmentation" / "yolo11m-road-seg.pt"

# ---------------------------------------------------------------------------
# Inference tuning – ALL modules run every frame (no skipping)
# ---------------------------------------------------------------------------
YOLO_INPUT_SIZE      = 320
DEPTH_INPUT_SIZE     = 256
FLOW_MAX_CORNERS     = 200
FLOW_DRAW_MAX_POINTS = 120
POTHOLE_INPUT_SIZE   = 416
POTHOLE_CONFIDENCE   = 0.50
ROAD_PATH_INPUT_SIZE = 640
ROAD_PATH_CONFIDENCE = 0.50
DISTANCE_MAX_OBJECTS = 12

# ---------------------------------------------------------------------------
# Depth conversion constants
# ---------------------------------------------------------------------------
DEPTH_NEAR_M = 0.4
DEPTH_FAR_M  = 10.0

# ---------------------------------------------------------------------------
# Adaptive temporal smoother constants (REMOVED)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Depth-region HUD constants
# ---------------------------------------------------------------------------
REGION_CLEAR_M    = 4.0
REGION_BLOCKED_M  = 2.0
WALL_COVERED_FRAC = 0.60
WALL_DEPTH_M      = 1.8

# ---------------------------------------------------------------------------
# Navigation Assistant constants
# ---------------------------------------------------------------------------
NAV_COOLDOWN_SECONDS       = 2.0
NAV_OBSTACLE_DISTANCE_M    = 3.5
NAV_CENTRE_BAND            = 0.15
NAV_POTHOLE_AREA_RATIO     = 0.03
NAV_WALKABLE_LOW_THRESHOLD = 30.0
NAV_WALKABLE_MED_THRESHOLD = 55.0
NAV_SM_REPEAT_DELAY_SECONDS = 5.0
NAV_SM_STATE_HOLD_FRAMES    = 3

# ---------------------------------------------------------------------------
# Label / object-class risk constants
# ---------------------------------------------------------------------------
PRESERVED_LABELS = {"person", "human", "car", "dog", "pothole"}
LABEL_PRIORITY   = ["human", "person", "car", "dog", "pothole", "obstacle"]

OBJECT_CLASS_RISK = {
    "human":    1.0,
    "person":   1.0,
    "car":      0.9,
    "dog":      0.75,
    "pothole":  0.8,
    "obstacle": 0.5,
}

# ---------------------------------------------------------------------------
# Optical flow thresholds
# ---------------------------------------------------------------------------
FLOW_FAST_THRESHOLD = 8.0   # px/frame → high approach speed
FLOW_MED_THRESHOLD  = 3.0   # px/frame → moderate motion

# ---------------------------------------------------------------------------
# NEW – Directional Safety Field constants
# ---------------------------------------------------------------------------
# Five fan-shaped regions emanating from the camera viewpoint
DIRECTION_NAMES = ["left", "slight_left", "forward", "slight_right", "right"]

# Angular boundaries (degrees from image centre, 0 = straight ahead)
# Total FOV ≈ 120° (typical webcam) split into 5 unequal sectors
DIR_ANGLE_BOUNDARIES = [-60.0, -30.0, -12.0, 12.0, 30.0, 60.0]

# Perspective-aware fan: how far up the frame (Y fraction) to extend
FAN_TOP_Y_FRAC    = 0.20   # top of fan region (far field)
FAN_BOTTOM_Y_FRAC = 1.00   # bottom of fan region (near field)
FAN_ORIGIN_Y_FRAC = 0.55   # vanishing-point Y (perspective origin)

# Risk weights for combining signals
RISK_W_DEPTH       = 0.35
RISK_W_OBJECT      = 0.25
RISK_W_FLOW        = 0.25
RISK_W_WALKABILITY = 0.15

# Temporal smoothing (EMA alpha for risk field)
RISK_TEMPORAL_ALPHA = 0.35   # lower = smoother, higher = more responsive

# Direction transition smoothing
DIR_SWITCH_HYSTERESIS   = 0.04  # minimum risk advantage to switch direction
DIR_SWITCH_HOLD_FRAMES  = 2     # minimum frames before direction can switch

# Emergency override thresholds
EMERGENCY_STOP_RISK   = 0.90   # forward risk above this → emergency steering/STOP
EMERGENCY_ALL_BLOCKED = 0.80   # if ALL directions above this → STOP

# Free-space depth threshold for a direction to be "passable"
FREE_SPACE_DEPTH_M = 1.0

# Risk sampling resolution per direction (radial slices)
RISK_RADIAL_SLICES = 8

# ---------------------------------------------------------------------------
# Depth risk weights
# ---------------------------------------------------------------------------
DEPTH_CONF_WEIGHT = 0.30
OBJECT_WEIGHT     = 0.30
FLOW_WEIGHT       = 0.25
POSITION_WEIGHT   = 0.15

# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------
RISK_FAR_THRESHOLD  = 0.35
RISK_NEAR_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
depth_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO main model
yolo_model = YOLO("/models/yolo/yolov8x.pt")

# Depth model
depth_model = None
try:
    if str(DEPTH_REPO_DIR) not in sys.path:
        sys.path.append(str(DEPTH_REPO_DIR))
    DepthAnythingV2  = importlib.import_module("depth_anything_v2.dpt").DepthAnythingV2
    depth_model      = DepthAnythingV2(encoder="vitl")
    depth_state_dict = torch.load(str(DEPTH_CHECKPOINT_PATH), map_location=depth_device)
    depth_model.load_state_dict(depth_state_dict, strict=False)
    depth_model.to(depth_device)
    depth_model.eval()
    print(f"Depth model loaded on {depth_device}: {DEPTH_CHECKPOINT_PATH}")
except Exception as exc:
    depth_model = None
    print(f"Depth model not available: {exc}")

# Pothole model
pothole_model = None
try:
    pothole_model = YOLO(str(POTHOLE_MODEL_PATH))
    print(f"Pothole model loaded: {POTHOLE_MODEL_PATH}")
except Exception as exc:
    pothole_model = None
    print(f"Pothole model not available: {exc}")

# Road-path segmentation model
road_path_model = None
try:
    road_path_model = YOLO(str(ROAD_PATH_MODEL_PATH))
    print(f"Road-path model loaded: {ROAD_PATH_MODEL_PATH}")
except Exception as exc:
    road_path_model = None
    print(f"Road-path model not available: {exc}")

# ---------------------------------------------------------------------------
# Shared mutable navigation state (singleton)
# ---------------------------------------------------------------------------

@dataclass
class DirectionalRiskResult:
    """Result from the new Autonomous Navigation Safety Field system."""
    # Per-direction risk scores: {name: float}
    direction_risks:    dict = field(default_factory=dict)
    # Output metrics
    safest_direction:        str = "forward"
    steering_strength:       float = 0.0  # -1.0 (hard left) to 1.0 (hard right)
    lane_confidence:         float = 0.8
    collision_probability:   float = 0.0
    path_continuity_score:   float = 1.0
    future_risk_score:       float = 0.0
    
    # Internal state & visualization
    is_emergency_stop:  bool = False
    nav_command:        str = "Forward"
    nav_angle_deg:      float = 0.0
    scene_risk:         float = 0.0
    confidence:         float = 0.8
    risk_heatmap:       Optional[np.ndarray] = None
    object_risks:       list = field(default_factory=list)
    direction_free_space: dict = field(default_factory=dict)


@dataclass
class UnifiedDetectionResult:
    primary_action:    str
    urgency:           str
    closest_threat:    str
    threat_distance_m: float
    threat_zone:       str
    threat_side:       str
    pothole_present:   bool
    walkable_pct:      float
    road_direction:    str
    confidence:        float
    # Directional safety extras
    scene_risk:            float = 0.0
    nav_angle_deg:         float = 0.0
    nav_command:           str = "Forward"
    direction_risks:       dict = field(default_factory=dict)
    # Car-like navigation outputs
    safest_direction:      str = "forward"
    steering_strength:     float = 0.0
    lane_confidence:       float = 0.8
    collision_probability: float = 0.0
    path_continuity_score: float = 1.0
    future_risk_score:     float = 0.0


class SharedNavState:
    road_direction:      str   = "STRAIGHT"
    walkable_pct:        float = 100.0
    pothole_detected:    bool  = False
    pothole_area_ratio:  float = 0.0
    directional_result:  Optional[DirectionalRiskResult] = None
    unified_detection:   Optional[UnifiedDetectionResult] = None
    segmentation_mask:   Optional[np.ndarray] = None


shared_nav_state = SharedNavState()
