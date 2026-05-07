# ---------------------------------------------------------------------------
# depth.py – Depth estimation pipeline (MiDaS / Depth-Anything-V2)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from config import (
    depth_model, depth_device, DEPTH_INPUT_SIZE,
    DEPTH_NEAR_M, DEPTH_FAR_M,
    REGION_CLEAR_M, REGION_BLOCKED_M, WALL_COVERED_FRAC, WALL_DEPTH_M,
)


# ===========================================================================
# Core inference
# ===========================================================================

def infer_depth_f32(frame_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Run depth model and return normalised float32 depth map [0,1]."""
    if depth_model is None:
        return np.full((h, w), 0.5, dtype=np.float32)
    try:
        with torch.no_grad():
            raw = depth_model.infer_image(frame_bgr, DEPTH_INPUT_SIZE)
        mn, mx = float(raw.min()), float(raw.max())
        span   = mx - mn if (mx - mn) > 1e-8 else 1e-8
        norm   = ((raw - mn) / span).astype(np.float32)
        if norm.shape != (h, w):
            norm = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
        return norm
    except Exception as exc:
        print(f"Depth inference error: {exc}")
        return np.full((h, w), 0.5, dtype=np.float32)


def depth_to_metres(norm_val: float) -> float:
    """Convert normalised depth [0,1] to estimated metres."""
    return float(DEPTH_FAR_M - norm_val * (DEPTH_FAR_M - DEPTH_NEAR_M))


def run_depth_map(frame_bgr: np.ndarray) -> np.ndarray:
    """Return depth as uint8 (0-255)."""
    h, w = frame_bgr.shape[:2]
    return (infer_depth_f32(frame_bgr, h, w) * 255.0).astype(np.uint8)


def run_depth_inference(frame_bgr: np.ndarray) -> np.ndarray:
    """Return colourised depth frame for display."""
    try:
        return cv2.applyColorMap(run_depth_map(frame_bgr), cv2.COLORMAP_INFERNO)
    except Exception as exc:
        print(f"Depth colourmap error: {exc}")
        return frame_bgr


# ===========================================================================
# Depth cache
# ===========================================================================

class DepthCache:
    """Per-frame depth cache."""
    def __init__(self):
        self.depth_f32: Optional[np.ndarray] = None
        self.depth_u8:  Optional[np.ndarray] = None

    def update(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        self.depth_f32 = infer_depth_f32(frame_bgr, h, w)
        self.depth_u8 = (self.depth_f32 * 255.0).astype(np.uint8)

    def reset(self):
        self.depth_f32 = None
        self.depth_u8  = None


# ===========================================================================
# Depth region analysis
# ===========================================================================

@dataclass
class DepthRegionResult:
    left_m:     float
    centre_m:   float
    right_m:    float
    clearest:   str
    wall_ahead: bool
    suggestion: str


def analyse_depth_regions(depth_f32: np.ndarray) -> DepthRegionResult:
    h, w    = depth_f32.shape
    y_start = int(h * 0.40)
    nav_rgn = depth_f32[y_start:, :]
    third   = w // 3
    left_m   = depth_to_metres(float(np.mean(nav_rgn[:, :third])))
    centre_m = depth_to_metres(float(np.mean(nav_rgn[:, third: 2 * third])))
    right_m  = depth_to_metres(float(np.mean(nav_rgn[:, 2 * third:])))
    norm_thresh = 1.0 - WALL_DEPTH_M / DEPTH_FAR_M
    close_frac  = float(np.mean(nav_rgn >= norm_thresh))
    wall_ahead  = close_frac >= WALL_COVERED_FRAC
    region_d = {"left": left_m, "centre": centre_m, "right": right_m}
    clearest = max(region_d, key=lambda k: region_d[k])
    max_open = max(left_m, centre_m, right_m)
    if wall_ahead or max_open < REGION_BLOCKED_M:
        suggestion = "path blocked"
    elif clearest == "left"  and left_m  > REGION_CLEAR_M:
        suggestion = "move left"
    elif clearest == "right" and right_m > REGION_CLEAR_M:
        suggestion = "move right"
    else:
        suggestion = "straight"
    return DepthRegionResult(
        left_m=left_m, centre_m=centre_m, right_m=right_m,
        clearest=clearest, wall_ahead=wall_ahead, suggestion=suggestion,
    )
