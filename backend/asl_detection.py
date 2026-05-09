import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
ASL_MODEL_PATH = BASE_DIR / "models" / "asl" / "yolov8x1.pt"


def _normalize_label(label: str) -> str:
    """
    Convert model labels into something TTS can pronounce reasonably.
    For example: "open_hand" -> "open hand"
    """
    cleaned = label.strip()
    cleaned = cleaned.replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


class LazyASLRecognizer:
    """
    Loads the ASL YOLO model lazily (first inference), so server startup stays lighter.
    """

    def __init__(self):
        self._model: Optional[YOLO] = None

    def _ensure_model_loaded(self) -> YOLO:
        if self._model is not None:
            return self._model

        # Path may be relative depending on runtime cwd; use absolute path.
        self._model = YOLO(str(ASL_MODEL_PATH))
        return self._model

    def infer(self, frame_bgr: np.ndarray, imgsz: int = 320) -> Tuple[Optional[str], float]:
        """
        Returns: (label, confidence)
        """
        model = self._ensure_model_loaded()

        results = model(frame_bgr, verbose=False, imgsz=imgsz)
        if not results:
            return None, 0.0

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None, 0.0

        # Pick the highest-confidence detection.
        boxes = r0.boxes
        best_i = int(np.argmax(boxes.conf.cpu().numpy()))
        conf = float(boxes.conf[best_i].item())
        cls_id = int(boxes.cls[best_i].item())

        names = r0.names
        if isinstance(names, dict):
            raw_label = str(names.get(cls_id, cls_id))
        elif isinstance(names, list):
            raw_label = str(names[cls_id]) if 0 <= cls_id < len(names) else str(cls_id)
        else:
            raw_label = str(cls_id)

        return _normalize_label(raw_label), conf


@dataclass
class ASLDecision:
    nav_command: str
    confidence: float
    should_speak: bool


class ASLAssistant:
    """
    Stabilizes ASL predictions to avoid flicker:
    - requires consecutive frames with the same label
    - uses a cooldown between spoken words
    """

    def __init__(
        self,
        speak_cooldown_s: float = 1.5,
        hold_frames: int = 3,
        min_conf: float = 0.35,
        unknown_label: str = "ASL",
    ):
        self._last_spoken: str = ""
        self._last_spoken_at: float = 0.0

        self._candidate: str = ""
        self._candidate_frames: int = 0
        self._candidate_conf: float = 0.0

        self._speak_cooldown_s = speak_cooldown_s
        self._hold_frames = hold_frames
        self._min_conf = min_conf
        self._unknown_label = unknown_label

    def decide(self, label: Optional[str], confidence: float) -> ASLDecision:
        now = time.monotonic()

        nav_command = label if (label and confidence >= self._min_conf) else self._unknown_label
        nav_command = nav_command.strip()

        # Update candidate stability
        if nav_command == self._candidate:
            self._candidate_frames += 1
            self._candidate_conf = confidence
        else:
            self._candidate = nav_command
            self._candidate_frames = 1
            self._candidate_conf = confidence

        stable = self._candidate_frames >= self._hold_frames
        cooldown_ok = (now - self._last_spoken_at) >= self._speak_cooldown_s
        changed = nav_command != self._last_spoken

        should_speak = stable and cooldown_ok and changed and nav_command != self._unknown_label
        if should_speak:
            self._last_spoken = nav_command
            self._last_spoken_at = now

        return ASLDecision(
            nav_command=nav_command,
            confidence=float(self._candidate_conf),
            should_speak=should_speak,
        )


asl_recognizer = LazyASLRecognizer()
asl_assistant = ASLAssistant()

