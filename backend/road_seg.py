# ---------------------------------------------------------------------------
# road_seg.py – Road/path segmentation
# ---------------------------------------------------------------------------
from typing import Optional

import cv2
import numpy as np

from config import (
    road_path_model, ROAD_PATH_INPUT_SIZE, ROAD_PATH_CONFIDENCE,
    NAV_CENTRE_BAND,
)


def compute_path_direction(mask_u8: np.ndarray) -> str:
    """Determine path direction from walkable mask."""
    moments = cv2.moments(mask_u8)
    if moments["m00"] == 0: return "STOP"
    _, width = mask_u8.shape
    cx = int(moments["m10"] / moments["m00"])
    offset = cx - width // 2
    if abs(offset) < width * NAV_CENTRE_BAND: return "STRAIGHT"
    return "RIGHT" if offset > 0 else "LEFT"


def run_road_path_inference(frame_bgr: np.ndarray):
    """
    Returns (annotated_frame, direction_str, walkable_pct).
    """
    if road_path_model is None:
        return frame_bgr, "STRAIGHT", 100.0
    try:
        results = road_path_model.predict(
            frame_bgr, imgsz=ROAD_PATH_INPUT_SIZE,
            conf=ROAD_PATH_CONFIDENCE, verbose=False)
        if not results: return frame_bgr, "STRAIGHT", 100.0
        result        = results[0]
        height, width = frame_bgr.shape[:2]
        if (result.masks is not None and result.masks.data is not None
                and len(result.masks.data) > 0):
            masks    = result.masks.data.detach().cpu().numpy()
            combined = np.any(masks, axis=0).astype(np.uint8)
            walkable = 1 - combined
        else:
            walkable = np.ones((height, width), dtype=np.uint8)
        if walkable.shape[:2] != (height, width):
            walkable = cv2.resize(walkable, (width, height),
                                  interpolation=cv2.INTER_NEAREST)
        overlay     = frame_bgr.copy()
        green_layer = np.zeros_like(overlay)
        green_layer[:] = (0, 255, 0)
        blended     = cv2.addWeighted(overlay, 0.7, green_layer, 0.3, 0)
        overlay[walkable.astype(bool)] = blended[walkable.astype(bool)]
        direction    = compute_path_direction(walkable)
        walkable_pct = float(np.mean(walkable)) * 100.0
        cv2.putText(overlay,
                    f"Road Path: {direction} | Walkable: {walkable_pct:.0f}%",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return overlay, direction, walkable_pct, walkable
    except Exception as exc:
        print(f"Road-path inference error: {exc}")
        return frame_bgr, "STRAIGHT", 100.0, None


