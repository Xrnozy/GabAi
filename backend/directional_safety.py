# ---------------------------------------------------------------------------
# directional_safety.py – Directional Safety Field (Core Navigation System)
# ---------------------------------------------------------------------------
"""
Replaces the old Far/Near/Close zone system with a perspective-aware
directional model using 5 fan-shaped regions:
    Left | Slight Left | Forward | Slight Right | Right

Each region's risk is computed continuously from:
    - Depth (distance → proximity risk)
    - Optical flow (approaching motion)
    - YOLO detections (object type/importance)
    - Segmentation (walkable vs blocked surface)

The risk field is temporally smoothed (EMA) for flicker-free output.
"""
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

from config import (
    DIRECTION_NAMES, DIR_ANGLE_BOUNDARIES,
    FAN_TOP_Y_FRAC, FAN_BOTTOM_Y_FRAC, FAN_ORIGIN_Y_FRAC,
    RISK_W_DEPTH, RISK_W_OBJECT, RISK_W_FLOW, RISK_W_WALKABILITY,
    RISK_TEMPORAL_ALPHA,
    DIR_SWITCH_HYSTERESIS, DIR_SWITCH_HOLD_FRAMES,
    EMERGENCY_STOP_RISK, EMERGENCY_ALL_BLOCKED,
    FREE_SPACE_DEPTH_M, RISK_RADIAL_SLICES,
    DEPTH_FAR_M, DEPTH_NEAR_M,
    FLOW_FAST_THRESHOLD, FLOW_MED_THRESHOLD,
    OBJECT_CLASS_RISK, NAV_CENTRE_BAND,
    WALL_DEPTH_M, WALL_COVERED_FRAC,
    DirectionalRiskResult,
)
from depth import depth_to_metres
from detection import normalize_label, label_priority, class_name_from_result


# ===========================================================================
# Fan-region geometry
# ===========================================================================

def build_fan_regions(frame_h: int, frame_w: int) -> Dict[str, np.ndarray]:
    """
    Build perspective-aware fan-shaped polygons for the 5 directional regions.
    
    Each region is a polygon that:
    - Originates from the vanishing point area (top-center of frame)
    - Fans outward toward the bottom of the frame
    - Is narrower at top (far field) and wider at bottom (near field)
    
    Returns dict mapping direction name → polygon points (Nx2 int32 array).
    """
    cx = frame_w / 2.0
    origin_y = int(frame_h * FAN_ORIGIN_Y_FRAC)
    top_y    = int(frame_h * FAN_TOP_Y_FRAC)
    bot_y    = int(frame_h * FAN_BOTTOM_Y_FRAC) - 1

    # Effective FOV scale: how many pixels per degree at bottom
    fov_scale_top = frame_w * 0.05  # narrower at top for slanted perspective
    fov_scale_bot = frame_w * 1.5   # wider at bottom for slanted perspective

    regions = {}
    for i, name in enumerate(DIRECTION_NAMES):
        left_angle  = DIR_ANGLE_BOUNDARIES[i]
        right_angle = DIR_ANGLE_BOUNDARIES[i + 1]

        # Convert angles to x-positions at top and bottom of frame
        # Top of fan (far field) – narrow
        top_left_x  = cx + (left_angle / 60.0)  * (fov_scale_top / 2.0)
        top_right_x = cx + (right_angle / 60.0) * (fov_scale_top / 2.0)

        # Bottom of fan (near field) – wide
        bot_left_x  = cx + (left_angle / 60.0)  * (fov_scale_bot / 2.0)
        bot_right_x = cx + (right_angle / 60.0) * (fov_scale_bot / 2.0)

        # Clamp to frame bounds
        top_left_x  = max(0, min(frame_w - 1, int(top_left_x)))
        top_right_x = max(0, min(frame_w - 1, int(top_right_x)))
        bot_left_x  = max(0, min(frame_w - 1, int(bot_left_x)))
        bot_right_x = max(0, min(frame_w - 1, int(bot_right_x)))

        # Polygon: top-left → top-right → bottom-right → bottom-left
        poly = np.array([
            [top_left_x,  top_y],
            [top_right_x, top_y],
            [bot_right_x, bot_y],
            [bot_left_x,  bot_y],
        ], dtype=np.int32)

        regions[name] = poly

    return regions


def point_in_direction(x: float, y: float, frame_h: int, frame_w: int) -> str:
    """Determine which directional region a point falls into."""
    cx = frame_w / 2.0
    # Normalise x position relative to center (-1 to 1)
    rel_x = (x - cx) / (frame_w / 2.0)

    # Compute effective angle based on y position (perspective)
    y_frac = y / frame_h
    # Higher y = closer to camera = wider effective angle
    perspective_factor = 0.3 + 0.7 * y_frac  # range [0.3, 1.0]
    effective_angle = rel_x * 60.0 / perspective_factor

    for i, name in enumerate(DIRECTION_NAMES):
        if DIR_ANGLE_BOUNDARIES[i] <= effective_angle < DIR_ANGLE_BOUNDARIES[i + 1]:
            return name

    # Fallback: extreme edges
    if effective_angle < DIR_ANGLE_BOUNDARIES[0]:
        return "left"
    return "right"


# ===========================================================================
# Per-direction risk computation
# ===========================================================================

def compute_direction_depth_risk(
    direction: str,
    mask: np.ndarray,
    full_depth_m: Optional[np.ndarray],
    frame_h: int,
) -> Tuple[float, float, float, float]:
    """
    Multi-Zone Depth Risk: Computes risks for Close (<0.5m), Near (0.5-2.0m), and Far (2.0-5.0m) zones.
    Returns (total_risk, close_risk, near_risk, free_space_metres).
    """
    if full_depth_m is None:
        return 0.5, 0.0, 0.0, 5.0

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.5, 0.0, 0.0, 5.0

    depths_m = full_depth_m[ys, xs]
    depths_m = depths_m[np.isfinite(depths_m)]
    depths_m = depths_m[depths_m > 0.0]
    if depths_m.size == 0:
        return 0.5, 0.0, 0.0, 5.0

    close_thresh = max(0.6, DEPTH_NEAR_M * 0.6)
    near_thresh = max(close_thresh + 0.2, DEPTH_NEAR_M)
    far_thresh = max(near_thresh + 0.5, DEPTH_FAR_M)

    # Occupancy-like risk (fraction of region with close/near depths).
    close_frac = float(np.mean(depths_m <= close_thresh))
    near_frac = float(np.mean((depths_m > close_thresh) & (depths_m <= near_thresh)))

    # Nearest-distance risk using robust low percentiles (avoids single-pixel noise).
    d05 = float(np.percentile(depths_m, 5))
    d20 = float(np.percentile(depths_m, 20))

    def proximity_risk(dist_m: float) -> float:
        if dist_m <= close_thresh:
            return 1.0
        if dist_m >= far_thresh:
            return 0.0
        return float(np.clip((far_thresh - dist_m) / (far_thresh - close_thresh), 0.0, 1.0))

    nearest_risk = max(proximity_risk(d05), 0.85 * proximity_risk(d20))
    occupancy_risk = np.clip((close_frac * 1.35) + (near_frac * 0.75), 0.0, 1.0)

    # Depth zones along image Y (far / near / close) should dominate safety.
    y_frac = ys.astype(np.float32) / max(1.0, float(frame_h - 1))
    zone_far = depths_m[y_frac < 0.45]
    zone_near = depths_m[(y_frac >= 0.45) & (y_frac < 0.72)]
    zone_close = depths_m[y_frac >= 0.72]

    def zone_score(zone_depths: np.ndarray) -> float:
        if zone_depths.size == 0:
            return 0.0
        z20 = float(np.percentile(zone_depths, 20))
        z_close = float(np.mean(zone_depths <= close_thresh))
        z_near = float(np.mean((zone_depths > close_thresh) & (zone_depths <= near_thresh)))
        return float(np.clip((0.7 * proximity_risk(z20)) + (0.3 * np.clip(1.1 * z_close + 0.6 * z_near, 0.0, 1.0)), 0.0, 1.0))

    zone_priority_risk = (
        0.60 * zone_score(zone_close) +
        0.30 * zone_score(zone_near) +
        0.10 * zone_score(zone_far)
    )

    # Blend: prioritize zone safety first, keep occupancy + nearest as stabilizers.
    total_risk = float(np.clip(
        (0.58 * zone_priority_risk) + (0.27 * nearest_risk) + (0.15 * occupancy_risk),
        0.0, 1.0
    ))

    close_risk = float(np.clip(close_frac, 0.0, 1.0))
    near_risk = float(np.clip(close_frac + near_frac, 0.0, 1.0))

    # Use lower percentile (not mean) so free-space reflects nearest drivable clearance.
    free_space_m = d20

    return total_risk, close_risk, near_risk, free_space_m


def compute_direction_flow_risk(
    direction: str,
    flow_vectors: list,
    frame_h: int,
    frame_w: int,
) -> Tuple[float, float]:
    """
    Predictive Risk: Computes base motion risk and future collision risk (closing speed).
    Returns (flow_risk, future_risk).
    """
    if not flow_vectors:
        return 0.0, 0.0

    region_mags = []
    closing_speeds = []
    
    cx, cy = frame_w / 2.0, frame_h / 2.0
    for old_pt, new_pt in flow_vectors:
        ox, oy = old_pt.ravel()[:2]
        nx, ny = new_pt.ravel()[:2]
        mag = float(np.linalg.norm(np.array([nx, ny]) - np.array([ox, oy])))
        region_mags.append(mag)
        
        dist_old = np.linalg.norm(np.array([ox, oy]) - np.array([cx, cy]))
        dist_new = np.linalg.norm(np.array([nx, ny]) - np.array([cx, cy]))
        # Closing motion means points move toward the camera-center corridor.
        if dist_new < dist_old:
            closing_speeds.append(mag)

    if not region_mags:
        return 0.0, 0.0

    avg_mag = float(np.mean(region_mags))
    avg_closing = float(np.mean(closing_speeds)) if closing_speeds else 0.0

    flow_risk = min(1.0, avg_mag / FLOW_FAST_THRESHOLD)
    future_risk = min(1.0, avg_closing / (FLOW_FAST_THRESHOLD * 0.8))

    return flow_risk, future_risk


def compute_direction_object_risk(
    direction: str,
    mask: np.ndarray,
    boxes, result,
    full_depth_m: Optional[np.ndarray],
    frame_h: int,
    frame_w: int,
) -> Tuple[float, list]:
    """
    Returns (aggregated_risk, list_of_object_dicts).
    """
    if boxes is None:
        return 0.0, []

    objects_in_region = []
    for box in boxes:
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        if conf < 0.25:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_w - 1, int(x2)), min(frame_h - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        box_cx, box_cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
        box_cx = max(0, min(frame_w - 1, box_cx))
        box_cy = max(0, min(frame_h - 1, box_cy))
        
        if mask[box_cy, box_cx] == 0:
            continue

        class_id  = int(box.cls[0].item()) if box.cls is not None else -1
        raw_label = class_name_from_result(result, class_id)
        label     = normalize_label(raw_label)

        if full_depth_m is not None:
            roi = full_depth_m[y1:y2, x1:x2]
            if roi.size > 0:
                roi = roi[np.isfinite(roi)]
                roi = roi[roi > 0.0]
                dist_m = float(np.percentile(roi, 20)) if roi.size > 0 else 5.0
            else:
                dist_m = 5.0
        else:
            dist_m = 5.0

        class_risk = OBJECT_CLASS_RISK.get(label, 0.5)
        proximity_risk = max(0.0, 1.0 - (dist_m - 1.0) / (DEPTH_FAR_M - 1.0))
        obj_risk = class_risk * proximity_risk

        rel = (box_cx - frame_w / 2) / frame_w
        side = "centre" if abs(rel) < NAV_CENTRE_BAND else ("right" if rel > 0 else "left")

        objects_in_region.append({
            "label":     label,
            "dist_m":    dist_m,
            "side":      side,
            "box":       (x1, y1, x2, y2),
            "risk_score": float(np.clip(obj_risk, 0.0, 1.0)),
            "direction": direction,
            "priority":  label_priority(label),
        })

    if not objects_in_region:
        return 0.0, []

    objects_in_region.sort(key=lambda o: o["dist_m"])
    weights = [1.0 / (i + 1) for i in range(len(objects_in_region))]
    scores  = [o["risk_score"] for o in objects_in_region]
    agg_risk = float(np.average(scores, weights=weights))

    return float(np.clip(agg_risk, 0.0, 1.0)), objects_in_region


def compute_direction_walkability_risk(
    direction: str,
    mask: np.ndarray,
    segmentation_mask: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    """
    Road / Walkable Path Integration:
    Returns (walkability_risk, path_continuity_score, obstacle_density).
    """
    if segmentation_mask is None:
        return 0.0, 1.0, 0.0

    region_walkable = segmentation_mask[mask > 0]
    if region_walkable.size == 0:
        return 0.0, 1.0, 0.0

    walkable_frac = float(np.mean(region_walkable))
    unwalkable_frac = 1.0 - walkable_frac
    
    path_continuity = walkable_frac
    obstacle_density = unwalkable_frac * 0.5
    
    return float(np.clip(unwalkable_frac, 0.0, 1.0)), path_continuity, obstacle_density


# ===========================================================================
# Directional Safety Field Engine
# ===========================================================================

class DirectionalSafetyField:
    """
    Core navigation engine replacing the old zone system.
    
    Computes a continuous risk field across 5 directional fan regions,
    with temporal smoothing and stable direction selection.
    """

    def __init__(self):
        # Temporally smoothed risk per direction
        self._smoothed_risks: Dict[str, float] = {d: 0.0 for d in DIRECTION_NAMES}
        # Direction stability tracking
        self._current_direction: str = "forward"
        self._direction_hold_counter: int = 0
        # Risk heatmap (for visualization)
        self._risk_heatmap: Optional[np.ndarray] = None
        # Frame dimensions cache
        self._frame_h: int = 0
        self._frame_w: int = 0
        # Cached fan region polygons and masks
        self._fan_regions: Dict[str, np.ndarray] = {}
        self._fan_masks: Dict[str, np.ndarray] = {}
        self._blurred_masks: Dict[str, np.ndarray] = {}

    def _ensure_regions(self, frame_h: int, frame_w: int):
        """Rebuild fan regions if frame size changed."""
        if frame_h != self._frame_h or frame_w != self._frame_w:
            self._frame_h = frame_h
            self._frame_w = frame_w
            self._fan_regions = build_fan_regions(frame_h, frame_w)
            
            self._fan_masks = {}
            self._blurred_masks = {}
            for name, poly in self._fan_regions.items():
                mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                self._fan_masks[name] = mask
                
                f_mask = np.zeros((frame_h, frame_w), dtype=np.float32)
                cv2.fillPoly(f_mask, [poly], 1.0)
                self._blurred_masks[name] = cv2.GaussianBlur(f_mask, (45, 45), 0)

            self._risk_heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)

    @staticmethod
    def _compute_forward_wall_risk(
        full_depth_m: Optional[np.ndarray],
        forward_mask: Optional[np.ndarray],
    ) -> float:
        """
        Detect a wall-like obstruction ahead from depth:
        - significant forward-area coverage at close distance, and
        - relatively low depth variance (flat plane / wall signature).
        """
        if full_depth_m is None or forward_mask is None:
            return 0.0

        depths = full_depth_m[forward_mask > 0]
        depths = depths[np.isfinite(depths)]
        depths = depths[depths > 0.0]
        if depths.size < 40:
            return 0.0

        close_frac = float(np.mean(depths <= WALL_DEPTH_M))
        if close_frac < max(0.35, WALL_COVERED_FRAC * 0.65):
            return 0.0

        depth_std = float(np.std(depths))
        # Lower variance means more wall-like/planar obstacle.
        flatness = float(np.clip(1.0 - (depth_std / 1.4), 0.0, 1.0))
        coverage = float(np.clip((close_frac - 0.35) / 0.65, 0.0, 1.0))
        return float(np.clip((0.6 * coverage) + (0.4 * flatness), 0.0, 1.0))

    @staticmethod
    def _opposite_direction(direction: str) -> str:
        return {
            "left": "right",
            "slight_left": "slight_right",
            "forward": "forward",
            "slight_right": "slight_left",
            "right": "left",
        }.get(direction, "forward")

    def compute(
        self,
        frame_bgr: np.ndarray,
        depth_f32: Optional[np.ndarray],
        depth_map_u8: np.ndarray,
        flow_vectors: list,
        avg_flow_mag: float,
        boxes,
        result,
        segmentation_mask: Optional[np.ndarray],
        pothole_detected: bool = False,
        pothole_area_ratio: float = 0.0,
    ) -> DirectionalRiskResult:
        frame_h, frame_w = frame_bgr.shape[:2]
        self._ensure_regions(frame_h, frame_w)

        full_depth_m = None
        if depth_f32 is not None:
            full_depth_m = 0.5 + (depth_f32 * 9.5)

        directional_flows = {d: [] for d in DIRECTION_NAMES}
        if flow_vectors:
            for old_pt, new_pt in flow_vectors:
                ox, oy = old_pt.ravel()[:2]
                ox_int, oy_int = int(ox), int(oy)
                if 0 <= oy_int < frame_h and 0 <= ox_int < frame_w:
                    for d in DIRECTION_NAMES:
                        if self._fan_masks[d][oy_int, ox_int] > 0:
                            directional_flows[d].append((old_pt, new_pt))
                            break

        raw_risks: Dict[str, float] = {}
        free_spaces: Dict[str, float] = {}
        path_scores: Dict[str, float] = {}
        future_risks: Dict[str, float] = {}
        close_risks: Dict[str, float] = {}
        near_risks: Dict[str, float] = {}
        all_objects: list = []
        max_close_risk = 0.0

        for direction in DIRECTION_NAMES:
            mask = self._fan_masks[direction]

            # 1. Depth risk (Multi-Zone)
            depth_risk, close_r, near_r, free_space = compute_direction_depth_risk(
                direction, mask, full_depth_m, frame_h)
            free_spaces[direction] = free_space
            close_risks[direction] = close_r
            near_risks[direction] = near_r
            max_close_risk = max(max_close_risk, close_r)

            # 2. Predictive Flow risk
            flow_risk, future_risk = compute_direction_flow_risk(
                direction, directional_flows[direction], frame_h, frame_w)
            future_risks[direction] = future_risk

            # 3. Object risk
            obj_risk, objects = compute_direction_object_risk(
                direction, mask, boxes, result, full_depth_m, frame_h, frame_w)
            all_objects.extend(objects)

            # 4. Walkability risk & Path Corridor
            walk_risk, continuity, obs_density = compute_direction_walkability_risk(
                direction, mask, segmentation_mask)
            path_scores[direction] = continuity

            # Weighted combination
            fan_signal_risk = (
                RISK_W_DEPTH * depth_risk
              + RISK_W_FLOW * flow_risk
              + RISK_W_OBJECT * obj_risk
              + RISK_W_WALKABILITY * walk_risk
              + 0.1 * obs_density
              + 0.15 * future_risk
            )
            # Priority to depth-zone danger (close/near/far), keep fan fusion as secondary.
            zone_priority_signal = np.clip((0.68 * close_r) + (0.32 * near_r), 0.0, 1.0)
            combined_risk = (0.62 * zone_priority_signal) + (0.38 * fan_signal_risk)
            raw_risks[direction] = float(np.clip(combined_risk, 0.0, 1.0))

        # Extra directional avoidance pressure from close/near objects.
        # If an object is close on one side, raise risk on that side and nudge opposite side safer.
        side_avoidance_bias = {d: 0.0 for d in DIRECTION_NAMES}
        for obj in all_objects:
            d = obj.get("direction")
            if d not in side_avoidance_bias:
                continue
            dist_m = float(obj.get("dist_m", DEPTH_FAR_M))
            score = float(obj.get("risk_score", 0.0))
            if dist_m <= 1.2:
                severity = 1.0
            elif dist_m <= 2.2:
                severity = 0.6
            else:
                severity = 0.0
            if severity <= 0.0:
                continue
            side_avoidance_bias[d] += severity * (0.18 + 0.22 * score)

        # Apply side/opposite coupling so the planner steers away from obstacle side.
        for d, bump in side_avoidance_bias.items():
            if bump <= 0.0:
                continue
            raw_risks[d] = float(np.clip(raw_risks[d] + bump, 0.0, 1.0))
            opp = self._opposite_direction(d)
            if opp != d:
                raw_risks[opp] = float(np.clip(raw_risks[opp] - (0.45 * bump), 0.0, 1.0))

        # Forward wall detection from depth map (global override into forward risk).
        wall_risk = self._compute_forward_wall_risk(full_depth_m, self._fan_masks.get("forward"))
        if wall_risk > 0.0:
            raw_risks["forward"] = max(raw_risks.get("forward", 0.0), float(np.clip(0.70 + 0.30 * wall_risk, 0.0, 1.0)))

        # Pothole override (injects into forward)
        if pothole_detected and pothole_area_ratio >= 0.01:
            raw_risks["forward"] = max(raw_risks.get("forward", 0.0), 0.95)

        # ── Temporal smoothing with fast rise / controlled decay ───────────
        # React quickly when risk increases (safety-critical), decay a bit slower
        # to prevent flicker when detections fluctuate frame-to-frame.
        alpha_rise = 0.82
        alpha_fall = 0.40
        for direction in DIRECTION_NAMES:
            old_val = self._smoothed_risks[direction]
            new_val = raw_risks[direction]
            alpha = alpha_rise if new_val >= old_val else alpha_fall
            self._smoothed_risks[direction] = old_val + alpha * (new_val - old_val)

        smoothed = dict(self._smoothed_risks)

        # Hard direction vetoes: directions with close hazards become non-selectable.
        # This prevents "turn into obstacle" behavior under noisy path bonuses.
        blocked_dirs = set()
        for d in DIRECTION_NAMES:
            if close_risks.get(d, 0.0) >= 0.52:
                blocked_dirs.add(d)
            elif near_risks.get(d, 0.0) >= 0.72 and future_risks.get(d, 0.0) >= 0.55:
                blocked_dirs.add(d)

        # ── Driving-Style Direction Scoring & Lane Memory ─────────────────
        steer_penalties = {
            "left": 0.12, "slight_left": 0.04, "forward": 0.0,
            "slight_right": 0.04, "right": 0.12
        }
        
        best_score = float('inf')
        recommended = self._current_direction
        
        for d in DIRECTION_NAMES:
            # Score = Base Risk + Steering Penalty - Path Bonus + Lane Memory
            lane_bonus = -0.03 if d == self._current_direction else 0.0
            forward_bonus = -0.02 if d == "forward" else 0.0
            veto_penalty = 2.0 if d in blocked_dirs else 0.0
            
            score = smoothed[d] + steer_penalties[d] - (path_scores[d] * 0.1) + lane_bonus + forward_bonus + veto_penalty
            
            if score < best_score:
                best_score = score
                recommended = d
        
        # Hysteresis: only commit to a new direction after hold frames
        if recommended != self._current_direction:
            self._direction_hold_counter += 1
            if self._direction_hold_counter >= DIR_SWITCH_HOLD_FRAMES:
                # Check hysteresis threshold
                current_score = smoothed[self._current_direction] + steer_penalties.get(self._current_direction, 0)
                new_score = smoothed[recommended] + steer_penalties.get(recommended, 0)
                if (current_score - new_score) > DIR_SWITCH_HYSTERESIS:
                    self._current_direction = recommended
                self._direction_hold_counter = 0
        else:
            self._direction_hold_counter = 0
            
        recommended = self._current_direction

        # ── Advanced Emergency Steering Logic ──────────────────────────────
        is_emergency = False
        nav_command = self._direction_to_command(recommended)
        
        fwd_risk = smoothed.get("forward", 0)
        
        # If forward is critical, try emergency steering instead of stopping immediately
        if fwd_risk >= EMERGENCY_STOP_RISK:
            # Check side corridors
            side_corridors = {d: smoothed[d] for d in ["slight_left", "slight_right", "left", "right"]}
            safest_side = min(side_corridors, key=side_corridors.get)
            
            if side_corridors[safest_side] < EMERGENCY_ALL_BLOCKED:
                recommended = safest_side
                nav_command = self._direction_to_command(safest_side)
            else:
                # All blocked -> Emergency Stop
                is_emergency = True
                nav_command = "Stop"
                recommended = "forward"
                
        # Fast moving obstacle at close range override
        if max_close_risk > 0.8 and max(future_risks.values()) > 0.7:
             is_emergency = True
             nav_command = "Stop"
        
        # TTC-like override: close + high approaching motion means imminent collision.
        for d in DIRECTION_NAMES:
            c = close_risks.get(d, 0.0)
            n = near_risks.get(d, 0.0)
            f = future_risks.get(d, 0.0)
            ttc_proxy = (0.65 * c) + (0.25 * n) + (0.10 * f)
            if (d == "forward" and ttc_proxy >= 0.72 and f >= 0.45) or (ttc_proxy >= 0.82 and f >= 0.55):
                is_emergency = True
                nav_command = "Stop"
                recommended = "forward"
                break
        
        # Depth wall emergency override if a wall blocks most of forward path.
        if wall_risk >= 0.78:
            is_emergency = True
            nav_command = "Stop"
            recommended = "forward"

        # Close vulnerable-object emergency override.
        # If a person/dog is very close in forward/near-front corridors, stop immediately.
        vulnerable_labels = {"human", "person", "dog"}
        for obj in all_objects:
            label = str(obj.get("label", "")).lower()
            if label not in vulnerable_labels:
                continue
            dist_m = float(obj.get("dist_m", DEPTH_FAR_M))
            direction = obj.get("direction", "forward")
            in_front_corridor = direction in {"forward", "slight_left", "slight_right"}
            # Slightly tighter rule for side corridors, stricter for direct forward.
            stop_thresh = 1.55 if direction == "forward" else 1.20
            if in_front_corridor and dist_m <= stop_thresh:
                is_emergency = True
                nav_command = "Stop"
                recommended = "forward"
                break

        # If planner still points to a blocked direction, force opposite safe corridor.
        if not is_emergency and recommended in blocked_dirs:
            opp = self._opposite_direction(recommended)
            if opp not in blocked_dirs:
                recommended = opp
            else:
                alternatives = [d for d in DIRECTION_NAMES if d not in blocked_dirs]
                if alternatives:
                    recommended = min(alternatives, key=lambda d: smoothed[d])
                else:
                    is_emergency = True
                    nav_command = "Stop"
                    recommended = "forward"
             
        # ── Autonomous Vehicle Navigation Output Metrics ───────────────────
        
        # Steering strength (-1.0 to 1.0)
        steer_map = {"left": -1.0, "slight_left": -0.5, "forward": 0.0, "slight_right": 0.5, "right": 1.0}
        steering_strength = steer_map.get(recommended, 0.0)
        
        # Collision probability (Max of forward risk and close risk)
        collision_prob = max(fwd_risk, max_close_risk)
        
        # Lane confidence
        lane_confidence = 1.0 - smoothed[recommended]
        
        # Scene risk aggregate
        dir_weights = {"left": 0.10, "slight_left": 0.15, "forward": 0.50, "slight_right": 0.15, "right": 0.10}
        scene_risk = sum(smoothed[d] * dir_weights[d] for d in DIRECTION_NAMES)
        scene_risk = float(np.clip(scene_risk, 0.0, 1.0))

        # Build risk heatmap with smooth gradients and corridors
        new_heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
        for direction in DIRECTION_NAMES:
            vis_risk = smoothed[direction]
            if path_scores.get(direction, 0.0) > 0.7:
                vis_risk *= 0.8
            new_heatmap += self._blurred_masks[direction] * vis_risk
        
        self._risk_heatmap = new_heatmap

        confidence = 0.8
        if depth_f32 is not None:
            local_var   = float(np.std(depth_f32))
            confidence  = float(np.clip(1.0 - local_var * 0.5, 0.3, 1.0))

        all_objects.sort(key=lambda o: (-o["risk_score"], o["dist_m"]))
        
        angle_map = {"left": -45.0, "slight_left": -20.0, "forward": 0.0, "slight_right": 20.0, "right": 45.0}

        return DirectionalRiskResult(
            direction_risks=smoothed,
            safest_direction=recommended,
            steering_strength=steering_strength,
            lane_confidence=lane_confidence,
            collision_probability=collision_prob,
            path_continuity_score=path_scores.get(recommended, 0.0),
            future_risk_score=future_risks.get(recommended, 0.0),
            
            nav_command=nav_command,
            nav_angle_deg=angle_map.get(recommended, 0.0),
            is_emergency_stop=is_emergency,
            risk_heatmap=self._risk_heatmap,
            object_risks=all_objects,
            scene_risk=scene_risk,
            confidence=confidence,
            direction_free_space=free_spaces,
        )


    @staticmethod
    def _direction_to_command(direction: str) -> str:
        """Convert internal direction name to user-facing navigation command."""
        return {
            "left":         "Left",
            "slight_left":  "Slight Left",
            "forward":      "Forward",
            "slight_right": "Slight Right",
            "right":        "Right",
        }.get(direction, "Forward")

    def get_fan_regions(self) -> Dict[str, np.ndarray]:
        """Get the cached fan region polygons (for visualization)."""
        return self._fan_regions

    def reset(self):
        """Reset the safety field state."""
        self._smoothed_risks = {d: 0.0 for d in DIRECTION_NAMES}
        self._current_direction = "forward"
        self._direction_hold_counter = 0
        self._risk_heatmap = None


# ===========================================================================
# Module-level singleton
# ===========================================================================
safety_field = DirectionalSafetyField()
