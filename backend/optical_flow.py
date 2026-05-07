# ---------------------------------------------------------------------------
# optical_flow.py – Optical flow tracking with improved stability
# ---------------------------------------------------------------------------
from typing import List, Tuple, Optional

import cv2
import numpy as np

from config import FLOW_MAX_CORNERS, FLOW_DRAW_MAX_POINTS, FLOW_MED_THRESHOLD, FLOW_FAST_THRESHOLD


# ===========================================================================
# Basic flow computation
# ===========================================================================

def compute_flow_vectors(prev_gray: np.ndarray, curr_gray: np.ndarray) -> list:
    """Compute optical flow vectors between consecutive frames."""
    features = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=FLOW_MAX_CORNERS,
        qualityLevel=0.02, minDistance=7, blockSize=7)
    if features is None: return []
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, features, None,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if next_points is None or status is None: return []
    good_new = next_points[status.reshape(-1) == 1]
    good_old = features[status.reshape(-1) == 1]
    return list(zip(good_old, good_new))


def compute_avg_flow_magnitude(flow_vectors: list) -> float:
    """Compute average flow magnitude from flow vector list."""
    if not flow_vectors:
        return 0.0
    mags = [float(np.linalg.norm(
                np.array(n).ravel() - np.array(o).ravel()))
            for o, n in flow_vectors]
    return float(np.mean(mags))


def run_optical_flow_inference(prev_gray: np.ndarray,
                               curr_gray: np.ndarray,
                               frame_bgr: np.ndarray) -> np.ndarray:
    """Standalone optical flow visualization for the flow video track."""
    overlay = frame_bgr.copy()
    try:
        features = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=FLOW_MAX_CORNERS,
            qualityLevel=0.02, minDistance=7, blockSize=7)
        if features is None: return overlay
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, features, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if next_points is None or status is None: return overlay
        good_new = next_points[status.reshape(-1) == 1]
        good_old = features[status.reshape(-1) == 1]
        for new_pt, old_pt in zip(good_new[:FLOW_DRAW_MAX_POINTS],
                                  good_old[:FLOW_DRAW_MAX_POINTS]):
            a, b = new_pt.ravel()
            c, d = old_pt.ravel()
            cv2.arrowedLine(overlay, (int(c), int(d)), (int(a), int(b)),
                            (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(overlay, (int(a), int(b)), 1, (0, 0, 255), -1)
        cv2.putText(overlay, f"Optical flow points: {len(good_new)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)
        return overlay
    except Exception as exc:
        print(f"Optical flow error: {exc}")
        return frame_bgr
