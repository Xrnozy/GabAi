# ---------------------------------------------------------------------------
# visualization.py – All overlay drawing for the directional safety field
# ---------------------------------------------------------------------------
from typing import Optional, Dict

import cv2
import numpy as np

from config import (
    DIRECTION_NAMES, DirectionalRiskResult,
    FLOW_MED_THRESHOLD, FLOW_FAST_THRESHOLD, FLOW_DRAW_MAX_POINTS,
    RISK_FAR_THRESHOLD, RISK_NEAR_THRESHOLD,
    REGION_CLEAR_M, REGION_BLOCKED_M,
)
from depth import DepthRegionResult


# ===========================================================================
# Colour palettes
# ===========================================================================

# Per-direction colours (hue-shifted for visual clarity)
DIRECTION_COLORS = {
    "left":         (255, 100, 50),    # blue-ish
    "slight_left":  (200, 180, 50),    # teal
    "forward":      (50, 220, 50),     # green
    "slight_right": (50, 180, 200),    # teal-yellow
    "right":        (50, 100, 255),    # orange-red
}

# Risk-to-colour gradient
def risk_to_color(risk: float) -> tuple:
    """Map risk [0,1] to BGR colour: green → yellow → orange → red."""
    if risk < 0.3:
        t = risk / 0.3
        return (0, int(220 * (1 - t * 0.3)), int(80 + 140 * t))  # green → yellow-green
    elif risk < 0.6:
        t = (risk - 0.3) / 0.3
        return (0, int(200 * (1 - t)), int(220 + 35 * t))  # yellow-green → orange
    else:
        t = (risk - 0.6) / 0.4
        return (0, int(60 * (1 - t)), int(180 + 75 * t))  # orange → red


# ===========================================================================
# Directional fan region visualization
# ===========================================================================

def draw_directional_fans(
    frame_bgr: np.ndarray,
    fan_regions: Dict[str, np.ndarray],
    direction_risks: Dict[str, float],
    safest_direction: str,
) -> np.ndarray:
    """
    Draw the 5 fan-shaped directional regions with risk-coloured fills.
    The safest direction gets a highlighted border.
    """
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]

    fan_layer = np.zeros_like(overlay)
    alpha_layer = np.zeros((h, w, 1), dtype=np.float32)

    # Draw filled fan regions with risk-based colours
    for direction in DIRECTION_NAMES:
        if direction not in fan_regions:
            continue
        poly = fan_regions[direction]
        risk = direction_risks.get(direction, 0.0)
        color = risk_to_color(risk)
        alpha = 0.12 + 0.08 * risk

        cv2.fillPoly(fan_layer, [poly], color)
        cv2.fillPoly(alpha_layer, [poly], float(alpha))

        # Border
        border_color = color
        border_thick = 1
        if direction == safest_direction:
            border_color = (0, 255, 180)  # bright cyan for safest
            border_thick = 3
        cv2.polylines(overlay, [poly], isClosed=True,
                      color=border_color, thickness=border_thick)

        # Label with risk percentage
        centroid = poly.mean(axis=0).astype(int)
        label_text = f"{direction.replace('_', ' ').title()}"
        risk_text  = f"{risk:.0%}"
        cv2.putText(overlay, label_text,
                    (centroid[0] - 30, centroid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, risk_text,
                    (centroid[0] - 15, centroid[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)

    overlay = (overlay * (1.0 - alpha_layer) + fan_layer * alpha_layer).astype(np.uint8)
    return overlay


# ===========================================================================
# Risk heatmap overlay
# ===========================================================================

def draw_risk_heatmap(
    frame_bgr: np.ndarray,
    risk_heatmap: Optional[np.ndarray],
) -> np.ndarray:
    """
    Overlay a smooth risk heatmap (green → red gradient) on the frame.
    """
    if risk_heatmap is None:
        return frame_bgr

    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]

    # Resize heatmap to frame size if needed
    if risk_heatmap.shape[:2] != (h, w):
        hm = cv2.resize(risk_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        hm = risk_heatmap

    # Convert to colourmap
    hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)

    if not hasattr(draw_risk_heatmap, "last_hm_u8") or not np.array_equal(hm_u8, draw_risk_heatmap.last_hm_u8):
        draw_risk_heatmap.last_hm_u8 = hm_u8
        draw_risk_heatmap.last_hm_colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    hm_colored = draw_risk_heatmap.last_hm_colored

    # Blend with low opacity
    cv2.addWeighted(hm_colored, 0.15, overlay, 0.85, 0, overlay)

    return overlay


# ===========================================================================
# Navigation arrow
# ===========================================================================

def draw_navigation_arrow(
    frame_bgr: np.ndarray,
    dir_result: DirectionalRiskResult,
) -> np.ndarray:
    """Draw a prominent navigation arrow pointing in the recommended direction."""
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]

    arrow_cx = w // 2
    arrow_cy = h - 70
    arrow_len = 55

    if dir_result.is_emergency_stop or dir_result.nav_command == "Stop":
        # Draw STOP X
        cv2.line(overlay, (arrow_cx - 22, arrow_cy - 22),
                 (arrow_cx + 22, arrow_cy + 22), (0, 0, 255), 4)
        cv2.line(overlay, (arrow_cx + 22, arrow_cy - 22),
                 (arrow_cx - 22, arrow_cy + 22), (0, 0, 255), 4)
        # Background circle
        cv2.circle(overlay, (arrow_cx, arrow_cy), 32, (0, 0, 180), 2)
        cv2.putText(overlay, "STOP", (arrow_cx - 28, arrow_cy + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        angle_rad = np.deg2rad(dir_result.nav_angle_deg - 90)  # 0 = up
        tip_x = int(arrow_cx + arrow_len * np.cos(angle_rad))
        tip_y = int(arrow_cy + arrow_len * np.sin(angle_rad))

        # Arrow colour based on scene risk
        if dir_result.scene_risk < RISK_FAR_THRESHOLD:
            arrow_color = (0, 255, 100)   # green
        elif dir_result.scene_risk < RISK_NEAR_THRESHOLD:
            arrow_color = (0, 180, 255)   # orange
        else:
            arrow_color = (0, 80, 255)    # red

        # Draw arrow with glow effect
        cv2.arrowedLine(overlay, (arrow_cx, arrow_cy), (tip_x, tip_y),
                        (0, 0, 0), 6, tipLength=0.35)  # shadow
        cv2.arrowedLine(overlay, (arrow_cx, arrow_cy), (tip_x, tip_y),
                        arrow_color, 4, tipLength=0.35)

        cmd_label = dir_result.nav_command.upper()
        cv2.putText(overlay, cmd_label, (arrow_cx - 40, arrow_cy + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, arrow_color, 2, cv2.LINE_AA)

    return overlay


# ===========================================================================
# Object bounding boxes with risk scores
# ===========================================================================

def draw_object_risks(
    frame_bgr: np.ndarray,
    object_risks: list,
) -> np.ndarray:
    """Draw risk-scored bounding boxes for detected objects."""
    overlay = frame_bgr.copy()

    for obj in object_risks[:12]:  # limit to top 12 objects
        x1, y1, x2, y2 = obj["box"]
        risk_score = obj["risk_score"]
        direction  = obj.get("direction", "forward")

        # Color by risk level
        box_color = risk_to_color(risk_score)
        thickness = 3 if risk_score > 0.7 else 2

        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, thickness)

        # Risk bar above box
        bar_w = x2 - x1
        bar_h = 5
        bar_y = max(0, y1 - bar_h - 2)
        cv2.rectangle(overlay, (x1, bar_y), (x2, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * risk_score)
        cv2.rectangle(overlay, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), box_color, -1)

        # Label
        label_txt = f"{obj['label']} {obj['dist_m']:.1f}m R:{risk_score:.2f}"
        cv2.putText(overlay, label_txt,
                    (x1, max(18, y1 - bar_h - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_color, 1, cv2.LINE_AA)

    return overlay


# ===========================================================================
# Scene status banner
# ===========================================================================

def draw_scene_banner(
    frame_bgr: np.ndarray,
    dir_result: DirectionalRiskResult,
) -> np.ndarray:
    """Draw the scene risk banner and confidence indicator."""
    overlay = frame_bgr
    h, w = overlay.shape[:2]

    scene_risk_pct = int(dir_result.scene_risk * 100)

    if dir_result.is_emergency_stop:
        banner_color = (0, 0, 255)
        banner_txt   = f"EMERGENCY STOP  Risk:{scene_risk_pct}%"
    elif dir_result.scene_risk < RISK_FAR_THRESHOLD:
        banner_color = (0, 200, 0)
        banner_txt   = f"SAFE  Risk:{scene_risk_pct}%"
    elif dir_result.scene_risk < RISK_NEAR_THRESHOLD:
        banner_color = (0, 165, 255)
        banner_txt   = f"CAUTION  Risk:{scene_risk_pct}%"
    else:
        banner_color = (0, 0, 255)
        banner_txt   = f"DANGER  Risk:{scene_risk_pct}%"

    # Background strip
    strip = overlay.copy()
    cv2.rectangle(strip, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(strip, 0.6, overlay, 0.4, 0, overlay)

    cv2.putText(overlay, banner_txt, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, banner_color, 2, cv2.LINE_AA)

    # Confidence
    cv2.putText(overlay, f"Conf:{dir_result.confidence:.0%}",
                (w - 120, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    return overlay


# ===========================================================================
# Flow overlay (on fusion frame)
# ===========================================================================

def draw_flow_arrows(
    frame_bgr: np.ndarray,
    flow_vectors: list,
) -> np.ndarray:
    """Draw optical flow arrows on the frame."""
    overlay = frame_bgr.copy()
    if not flow_vectors:
        return overlay

    for old_pt, new_pt in list(flow_vectors)[:FLOW_DRAW_MAX_POINTS]:
        ox, oy = old_pt.ravel()[:2]
        nx, ny = new_pt.ravel()[:2]
        mag = float(np.linalg.norm(np.array([nx - ox, ny - oy])))
        if mag > FLOW_MED_THRESHOLD:
            color = (0, 255, 200) if mag < FLOW_FAST_THRESHOLD else (0, 100, 255)
            cv2.arrowedLine(overlay, (int(ox), int(oy)), (int(nx), int(ny)),
                            color, 1, tipLength=0.35)
    return overlay


# ===========================================================================
# Combined fusion overlay
# ===========================================================================

def draw_distance_zones(frame_bgr: np.ndarray) -> np.ndarray:
    """Draw 3 horizontal distance zones (FAR, NEAR, CLOSE) over the fan."""
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]

    y_close = int(h * 0.75)
    y_near  = int(h * 0.45)
    y_far   = int(h * 0.20)

    alpha_layer = np.zeros_like(overlay)
    cv2.rectangle(alpha_layer, (0, y_close), (w, h), (0, 0, 255), -1)
    cv2.rectangle(alpha_layer, (0, y_near), (w, y_close), (0, 165, 255), -1)
    cv2.rectangle(alpha_layer, (0, y_far), (w, y_near), (0, 255, 0), -1)

    overlay = cv2.addWeighted(alpha_layer, 0.1, overlay, 1.0, 0)

    for y, label, color in [
        (y_close, "CLOSE", (0, 0, 255)),
        (y_near,  "NEAR",  (0, 165, 255)),
        (y_far,   "FAR",   (0, 255, 0))
    ]:
        cv2.line(overlay, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.putText(overlay, label, (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return overlay


def draw_full_directional_overlay(
    frame_bgr: np.ndarray,
    dir_result: DirectionalRiskResult,
    fan_regions: Dict[str, np.ndarray],
    flow_vectors: list,
    avg_flow_mag: float,
) -> np.ndarray:
    """
    Complete directional safety field visualization pipeline.
    Layers: heatmap → fans → objects → arrow → banner → info bar.
    Flow arrows removed for cleaner output (flow still used in risk computation).
    """
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]

    # 1. Risk heatmap (subtle background)
    overlay = draw_risk_heatmap(overlay, dir_result.risk_heatmap)

    # 2. Directional fan regions
    overlay = draw_directional_fans(
        overlay, fan_regions,
        dir_result.direction_risks,
        dir_result.safest_direction)

    # 2.5 Distance zones
    overlay = draw_distance_zones(overlay)

    # 3. Object risk boxes
    overlay = draw_object_risks(overlay, dir_result.object_risks)

    # 4. Navigation arrow
    overlay = draw_navigation_arrow(overlay, dir_result)

    # 5. Scene banner (no CMD text)
    overlay = draw_scene_banner(overlay, dir_result)

    # 6. Info bar at bottom
    info_txt = (f"Safety Field | "
                f"Obj:{len(dir_result.object_risks)} "
                f"Flow:{avg_flow_mag:.1f}")
    cv2.putText(overlay, info_txt, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    return overlay


# ===========================================================================
# Depth region drawing (kept for depth video track)
# ===========================================================================

def draw_depth_regions(frame_bgr: np.ndarray, dr: DepthRegionResult) -> np.ndarray:
    overlay = frame_bgr.copy()
    h, w    = overlay.shape[:2]
    third   = w // 3
    y_bar   = h - 36

    def _colour(dist_m: float):
        if dist_m >= REGION_CLEAR_M:   return (0, 200, 0)
        if dist_m >= REGION_BLOCKED_M: return (0, 165, 255)
        return (0, 0, 255)

    panels = [
        (0,         third,     dr.left_m,   "L"),
        (third,     2 * third, dr.centre_m, "C"),
        (2 * third, w,         dr.right_m,  "R"),
    ]
    for x0, x1_, dist_m, lbl in panels:
        c = _colour(dist_m)
        cv2.rectangle(overlay, (x0, y_bar), (x1_, h), c, -1)
        cv2.rectangle(overlay, (x0, y_bar), (x1_, h), (255, 255, 255), 1)
        cv2.putText(overlay, f"{lbl}:{dist_m:.1f}m", (x0 + 4, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    if dr.wall_ahead:
        cv2.putText(overlay, "WALL AHEAD", (10, y_bar - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return overlay
