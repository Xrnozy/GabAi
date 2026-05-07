# ---------------------------------------------------------------------------
# navigation.py – Navigation assistant + unified detection
# ---------------------------------------------------------------------------
import json
import time
from typing import Optional

import numpy as np

from config import (
    UnifiedDetectionResult, DirectionalRiskResult,
    NAV_COOLDOWN_SECONDS, NAV_POTHOLE_AREA_RATIO,
    NAV_WALKABLE_LOW_THRESHOLD, NAV_WALKABLE_MED_THRESHOLD,
    RISK_FAR_THRESHOLD, RISK_NEAR_THRESHOLD,
    DIRECTION_NAMES,
)

from visualization import risk_to_color


# ===========================================================================
# Unified detection result builder
# ===========================================================================

def compute_unified_detection(
    dir_result: Optional[DirectionalRiskResult],
    pothole_detected: bool,
    pothole_area_ratio: float,
    road_direction: str,
    walkable_pct: float,
) -> UnifiedDetectionResult:
    """
    Build a unified detection result from the directional safety field.
    Replaces the old collision-zone-based logic.
    """
    primary_action    = "PATH_CLEAR"
    urgency           = "clear"
    closest_threat    = ""
    threat_distance_m = float("inf")
    threat_zone       = "none"
    threat_side       = "none"
    confidence        = dir_result.confidence if dir_result else 0.5
    scene_risk        = dir_result.scene_risk if dir_result else 0.0
    nav_angle         = dir_result.nav_angle_deg if dir_result else 0.0
    nav_command       = dir_result.nav_command if dir_result else "Forward"
    direction_risks   = dir_result.direction_risks if dir_result else {}
    
    # Car-like output metrics
    safest_direction      = dir_result.safest_direction if dir_result else "forward"
    steering_strength     = dir_result.steering_strength if dir_result else 0.0
    lane_confidence       = dir_result.lane_confidence if dir_result else 0.8
    collision_probability = dir_result.collision_probability if dir_result else 0.0
    path_continuity_score = dir_result.path_continuity_score if dir_result else 1.0
    future_risk_score     = dir_result.future_risk_score if dir_result else 0.0

    if dir_result:
        # Extract closest threat from object risks
        if dir_result.object_risks:
            top = dir_result.object_risks[0]
            closest_threat    = top["label"]
            threat_distance_m = top["dist_m"]
            threat_side       = top.get("side", "centre")
            threat_zone       = top.get("direction", "forward")

        # Priority 0: Emergency stop
        if dir_result.is_emergency_stop:
            urgency        = "critical"
            primary_action = "STOP"

        # Priority 1: High forward risk
        elif dir_result.direction_risks.get("forward", 0) >= RISK_NEAR_THRESHOLD:
            urgency = "urgent"
            cmd_map = {
                "Left": "MOVE_LEFT", "Slight Left": "MOVE_LEFT",
                "Right": "MOVE_RIGHT", "Slight Right": "MOVE_RIGHT",
                "Forward": "CAUTION", "Stop": "STOP",
            }
            primary_action = cmd_map.get(dir_result.nav_command, "CAUTION")

        # Priority 2: Moderate risk
        elif scene_risk >= RISK_FAR_THRESHOLD:
            urgency = "warning"
            cmd_map = {
                "Left": "MOVE_LEFT", "Slight Left": "MOVE_LEFT",
                "Right": "MOVE_RIGHT", "Slight Right": "MOVE_RIGHT",
                "Forward": "PATH_CLEAR", "Stop": "STOP",
            }
            primary_action = cmd_map.get(dir_result.nav_command, "PATH_CLEAR")

        # Priority 3: Low risk, path is clear
        else:
            urgency        = "clear"
            primary_action = "PATH_CLEAR"

    # Priority 4: Pothole override
    if pothole_detected and pothole_area_ratio >= NAV_POTHOLE_AREA_RATIO:
        if urgency in ("clear", "info"):
            closest_threat    = "pothole"
            threat_distance_m = 1.5
            threat_zone       = "forward"
            urgency           = "urgent"
            primary_action    = "CAUTION"

    # Priority 5: Walkability override
    if walkable_pct < NAV_WALKABLE_LOW_THRESHOLD or road_direction == "BLOCKED":
        if urgency in ("clear", "info", "warning"):
            urgency        = "critical"
            primary_action = "STOP"
    elif walkable_pct < NAV_WALKABLE_MED_THRESHOLD:
        if urgency in ("clear", "info"):
            urgency        = "warning"
            primary_action = "CAUTION"

    # Priority 6: Road direction guidance (only when path is clear)
    if primary_action in ("PATH_CLEAR", "CAUTION") and urgency in ("clear", "info"):
        if road_direction == "LEFT":
            primary_action = "MOVE_LEFT"
            urgency = "info"
        elif road_direction == "RIGHT":
            primary_action = "MOVE_RIGHT"
            urgency = "info"

    # Synchronize nav_command for WebRTC TTS if we must STOP
    if primary_action == "STOP":
        nav_command = "Stop"

    # Add distance zone to the nav command
    overall_distance = threat_distance_m
    if dir_result and dir_result.direction_free_space:
        min_free_space = min(dir_result.direction_free_space.values()) if dir_result.direction_free_space else 5.0
        overall_distance = min(overall_distance, min_free_space)

    distance_zone = "far"
    if overall_distance < 1.0:
        distance_zone = "close"
    elif overall_distance < 3.0:
        distance_zone = "near"

    if primary_action != "PATH_CLEAR":
        threat_name = closest_threat if closest_threat else "obstacle"
        nav_command = f"{nav_command}, {threat_name} {distance_zone}"

    return UnifiedDetectionResult(
        primary_action=primary_action,
        urgency=urgency,
        closest_threat=closest_threat,
        threat_distance_m=threat_distance_m,
        threat_zone=threat_zone,
        threat_side=threat_side,
        pothole_present=pothole_detected,
        walkable_pct=walkable_pct,
        road_direction=road_direction,
        confidence=confidence,
        scene_risk=scene_risk,
        nav_angle_deg=nav_angle,
        nav_command=nav_command,
        direction_risks=direction_risks,
        safest_direction=safest_direction,
        steering_strength=steering_strength,
        lane_confidence=lane_confidence,
        collision_probability=collision_probability,
        path_continuity_score=path_continuity_score,
        future_risk_score=future_risk_score,
    )


# ===========================================================================
# Navigation data sender (WebRTC data channel)
# ===========================================================================

def send_navigation_data(unified_result: UnifiedDetectionResult,
                         pcs: set,
                         should_speak: bool = False) -> None:
    """Send navigation data to all connected peers via data channel."""
    if not unified_result:
        return

    def _bgr_to_hex(bgr) -> str:
        b, g, r = [int(x) for x in bgr]
        return f"#{r:02x}{g:02x}{b:02x}"

    nav_data = {
        "primary":          unified_result.primary_action,
        "secondary":        unified_result.closest_threat or "No threats detected",
        "urgency":          unified_result.urgency,
        "path_direction":   unified_result.road_direction,
        "walkable_pct":     unified_result.walkable_pct if unified_result.walkable_pct is not None else 0,
        "threat_zone":      unified_result.threat_zone,
        "distance_m":       unified_result.threat_distance_m,
        "pothole_present":  unified_result.pothole_present,
        "should_speak":     should_speak,
        # Directional safety fields
        "scene_risk":       round(unified_result.scene_risk, 3),
        "nav_angle_deg":    round(unified_result.nav_angle_deg, 1),
        "nav_command":      unified_result.nav_command,
        "confidence":       round(unified_result.confidence, 3),
        "direction_risks":  {k: round(v, 3) for k, v in unified_result.direction_risks.items()},
        # Fan-zone colours (match backend/visualization.py risk_to_color exactly)
        # Order matches config.DIRECTION_NAMES: left → slight_left → forward → slight_right → right
        "fan_zone_colors": {
            d: _bgr_to_hex(risk_to_color(float(unified_result.direction_risks.get(d, 0.0))))
            for d in DIRECTION_NAMES
        },
        
        # Car-like navigation output metrics
        "safest_direction":      unified_result.safest_direction,
        "steering_strength":     round(unified_result.steering_strength, 2),
        "lane_confidence":       round(unified_result.lane_confidence, 3),
        "collision_probability": round(unified_result.collision_probability, 3),
        "path_continuity_score": round(unified_result.path_continuity_score, 3),
        "future_risk_score":     round(unified_result.future_risk_score, 3),
    }
    try:
        nav_json = json.dumps(nav_data)
    except Exception as e:
        print(f"[NAV] Serialize error: {e}")
        return
    for pc in list(pcs):
        if pc.connectionState != "connected":
            continue
        ch = getattr(pc, 'navigation_channel', None)
        if ch and ch.readyState == "open":
            try:
                ch.send(nav_json)
            except Exception as e:
                print(f"[NAV] Send error: {e}")


# ===========================================================================
# Navigation assistant (cooldown + logging)
# ===========================================================================

class NavigationAssistant:
    def __init__(self):
        self._last_state: str = "PATH_CLEAR"
        self._last_instruction: str = ""
        self._last_emit_time: float = 0.0
        self._state_hold_counter: int = 0

    def emit_unified(self, unified_result: UnifiedDetectionResult) -> Optional[str]:
        if not unified_result or unified_result.primary_action == "PATH_CLEAR":
            instruction = None
            current_state = "PATH_CLEAR"
        else:
            current_state = unified_result.primary_action
            if unified_result.closest_threat:
                instruction = (f"[{unified_result.nav_command.upper()}] "
                               f"{unified_result.primary_action} – "
                               f"{unified_result.closest_threat}")
            else:
                instruction = unified_result.primary_action

        now = time.monotonic()
        is_new_state = (current_state != self._last_state)

        if is_new_state:
            self._state_hold_counter += 1
            from config import NAV_SM_STATE_HOLD_FRAMES
            if self._state_hold_counter >= NAV_SM_STATE_HOLD_FRAMES:
                self._last_state = current_state
                self._state_hold_counter = 0
            else:
                return None  # Wait for state to stabilize
        else:
            self._state_hold_counter = 0

        # After stabilizing, self._last_state is the effective state
        if self._last_state == "PATH_CLEAR" or instruction is None:
            return None

        from config import NAV_COOLDOWN_SECONDS, NAV_SM_REPEAT_DELAY_SECONDS
        
        # Deduplicate identical strings within normal cooldown
        if instruction == self._last_instruction and (now - self._last_emit_time) < NAV_COOLDOWN_SECONDS:
            return None

        # Deduplicate identical states with a longer delay
        if not is_new_state and (now - self._last_emit_time) < NAV_SM_REPEAT_DELAY_SECONDS:
            return None

        self._last_instruction = instruction
        self._last_emit_time   = now
        self._emit(instruction, unified_result.urgency)
        return instruction

    @staticmethod
    def _emit(instruction: str, urgency: str = "info") -> None:
        icons = {"critical":"🚨","urgent":"⚠️","warning":"⚡","info":"ℹ️","clear":"✅"}
        print(f"[NAV {time.strftime('%H:%M:%S')}] {icons.get(urgency,'•')} {instruction}")


nav_assistant = NavigationAssistant()
