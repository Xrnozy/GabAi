import time
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Optional, Tuple

import numpy as np


# Simple, dependency-free color naming tuned for accessibility.
# (Avoids matplotlib/webcolors deps.)
_NAMED_RGB: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (220, 20, 60),
    "orange": (255, 140, 0),
    "yellow": (255, 215, 0),
    "green": (34, 139, 34),
    "cyan": (0, 180, 180),
    "blue": (30, 144, 255),
    "purple": (138, 43, 226),
    "pink": (255, 105, 180),
    "brown": (139, 69, 19),
}


def _rgb_to_hsv01(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [float(x) / 255.0 for x in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    v = mx
    d = mx - mn
    s = 0.0 if mx <= 1e-9 else d / mx
    return 0.0, s, v


def closest_color_name(rgb: Tuple[int, int, int]) -> Tuple[str, float]:
    """
    Returns (name, confidence) for an RGB tuple (0-255).
    Confidence is a rough [0, 1] score based on distance.
    """
    rgb = tuple(int(np.clip(x, 0, 255)) for x in rgb)  # type: ignore[assignment]

    _, s, v = _rgb_to_hsv01(rgb)
    # Quick rules for achromatic colors.
    if v < 0.15:
        return "black", 0.95
    if v > 0.92 and s < 0.18:
        return "white", 0.95
    if s < 0.16:
        return "gray", 0.8

    r, g, b = [x / 255.0 for x in rgb]
    min_dist = float("inf")
    closest = "unknown"
    for name, (cr, cg, cb) in _NAMED_RGB.items():
        rr, gg, bb = cr / 255.0, cg / 255.0, cb / 255.0
        dist = sqrt((r - rr) ** 2 + (g - gg) ** 2 + (b - bb) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest = name

    # Max possible distance in RGB cube is sqrt(3).
    confidence = float(np.clip(1.0 - (min_dist / sqrt(3.0)), 0.0, 1.0))
    return closest, confidence


@dataclass
class ColorDecision:
    name: str
    rgb: Tuple[int, int, int]
    confidence: float
    should_speak: bool


class ColorAssistant:
    """
    Stabilizes color predictions to avoid flicker/spam.
    """

    def __init__(
        self,
        speak_cooldown_s: float = 2.0,
        hold_frames: int = 2,
        min_conf: float = 0.25,
        unknown_label: str = "unknown",
    ):
        self._last_spoken: str = ""
        self._last_spoken_at: float = 0.0

        self._candidate: str = ""
        self._candidate_frames: int = 0
        self._candidate_rgb: Tuple[int, int, int] = (0, 0, 0)
        self._candidate_conf: float = 0.0

        self._speak_cooldown_s = speak_cooldown_s
        self._hold_frames = hold_frames
        self._min_conf = min_conf
        self._unknown_label = unknown_label

    def decide(self, name: Optional[str], rgb: Tuple[int, int, int], confidence: float) -> ColorDecision:
        now = time.monotonic()
        safe_name = (name or "").strip().lower() or self._unknown_label
        if confidence < self._min_conf:
            safe_name = self._unknown_label

        if safe_name == self._candidate:
            self._candidate_frames += 1
            self._candidate_rgb = rgb
            self._candidate_conf = confidence
        else:
            self._candidate = safe_name
            self._candidate_frames = 1
            self._candidate_rgb = rgb
            self._candidate_conf = confidence

        stable = self._candidate_frames >= self._hold_frames
        cooldown_ok = (now - self._last_spoken_at) >= self._speak_cooldown_s
        changed = safe_name != self._last_spoken
        should_speak = stable and cooldown_ok and changed and safe_name != self._unknown_label

        if should_speak:
            self._last_spoken = safe_name
            self._last_spoken_at = now

        return ColorDecision(
            name=self._candidate,
            rgb=self._candidate_rgb,
            confidence=float(self._candidate_conf),
            should_speak=bool(should_speak),
        )


color_assistant = ColorAssistant()

