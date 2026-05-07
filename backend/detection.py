# ---------------------------------------------------------------------------
# detection.py – YOLO detection, pothole inference, label utilities
# ---------------------------------------------------------------------------
from typing import Optional

import cv2
import numpy as np

from config import (
    yolo_model, pothole_model,
    YOLO_INPUT_SIZE, POTHOLE_INPUT_SIZE, POTHOLE_CONFIDENCE,
    PRESERVED_LABELS, LABEL_PRIORITY, OBJECT_CLASS_RISK,
    NAV_CENTRE_BAND,
)


# ===========================================================================
# Label utilities
# ===========================================================================

def normalize_label(raw_label: str) -> str:
    label = raw_label.lower().strip()
    if label == "person":  return "human"
    if label in PRESERVED_LABELS: return label
    return "obstacle"


def label_priority(label: str) -> int:
    try:    return LABEL_PRIORITY.index(label)
    except: return len(LABEL_PRIORITY)


def class_name_from_result(result, class_id: int) -> str:
    names = result.names
    if isinstance(names, dict):  return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


# ===========================================================================
# YOLO general detection
# ===========================================================================

def run_yolo_inference(frame_bgr: np.ndarray) -> np.ndarray:
    """Run YOLO and return annotated frame."""
    try:
        results = yolo_model(frame_bgr, verbose=False, imgsz=YOLO_INPUT_SIZE)
        if not results: return frame_bgr
        return results[0].plot()
    except Exception as exc:
        print(f"YOLO inference error: {exc}")
        return frame_bgr


def run_yolo_raw(frame_bgr: np.ndarray):
    """Run YOLO and return raw results (for fusion pipeline)."""
    try:
        results = yolo_model(frame_bgr, verbose=False, imgsz=YOLO_INPUT_SIZE)
        if not results:
            return None
        return results[0]
    except Exception as exc:
        print(f"YOLO raw inference error: {exc}")
        return None


# ===========================================================================
# Pothole detection
# ===========================================================================

def run_pothole_inference(frame_bgr: np.ndarray):
    """Returns (annotated_frame, detected_bool, max_area_ratio)."""
    if pothole_model is None:
        return frame_bgr, False, 0.0
    try:
        results = pothole_model.predict(
            frame_bgr, imgsz=POTHOLE_INPUT_SIZE,
            conf=POTHOLE_CONFIDENCE, verbose=False)
        if not results: return frame_bgr, False, 0.0
        overlay       = frame_bgr.copy()
        result        = results[0]
        boxes         = result.boxes
        pothole_count = 0
        height, width = frame_bgr.shape[:2]
        frame_area    = height * width
        max_area_ratio = 0.0
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                if conf < POTHOLE_CONFIDENCE: continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(width-1, int(x2)), min(height-1, int(y2))
                if x2 <= x1 or y2 <= y1: continue
                max_area_ratio = max(max_area_ratio, ((x2-x1)*(y2-y1))/frame_area)
                class_id   = int(box.cls[0].item()) if box.cls is not None else -1
                class_name = class_name_from_result(result, class_id)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(overlay, f"{class_name} {conf:.2f}",
                            (x1, max(18, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                pothole_count += 1
        cv2.putText(overlay, f"Pothole detections: {pothole_count}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return overlay, pothole_count > 0, max_area_ratio
    except Exception as exc:
        print(f"Pothole inference error: {exc}")
        return frame_bgr, False, 0.0
