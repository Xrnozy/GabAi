import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


class LazyTesseractRecognizer:
    """
    Tesseract recognizer for real-time OCR mode.
    Returns (text, confidence) where confidence is [0, 1].
    """

    def __init__(self):
        self._backend: Optional[str] = None
        self._pytesseract = None

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        try:
            import pytesseract  # type: ignore
            try:
                _ = pytesseract.get_tesseract_version()
            except Exception as exc:
                self._backend = "none"
                print(f"[TESSERACT] Tesseract not available: {exc}")
                return

            self._pytesseract = pytesseract
            self._backend = "pytesseract"
            print("[TESSERACT] pytesseract loaded")
        except Exception as exc:
            self._backend = "none"
            print(f"[TESSERACT] pytesseract load failed: {exc}")

    def infer(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        self._ensure_loaded()
        if self._backend != "pytesseract" or self._pytesseract is None:
            return None, 0.0

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            data = self._pytesseract.image_to_data(
                rgb,
                output_type=self._pytesseract.Output.DICT,
            )

            texts = []
            confs = []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                t = _clean_text(str(text))
                if not t:
                    continue
                texts.append(t)
                try:
                    c = float(conf)
                except Exception:
                    c = -1.0
                if c >= 0.0:
                    confs.append(c / 100.0)

            merged = _clean_text(" ".join(texts))
            if not merged:
                return None, 0.0
            confidence = float(np.clip(np.mean(confs), 0.0, 1.0)) if confs else 0.5
            return merged, confidence
        except Exception as exc:
            print(f"[TESSERACT] inference failed: {exc}")
            return None, 0.0


@dataclass
class OCRDecision:
    nav_command: str
    confidence: float
    should_speak: bool


class OCRAssistant:
    """
    Stabilizes OCR text so the app doesn't spam noisy frame-by-frame changes.
    """

    def __init__(
        self,
        speak_cooldown_s: float = 1.4,
        hold_frames: int = 2,
        min_conf: float = 0.25,
        unknown_label: str = "No text",
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

    def decide(self, text: Optional[str], confidence: float) -> OCRDecision:
        now = time.monotonic()
        nav_command = text if (text and confidence >= self._min_conf) else self._unknown_label
        nav_command = _clean_text(nav_command)

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

        return OCRDecision(nav_command=nav_command, confidence=float(self._candidate_conf), should_speak=bool(should_speak))


ocr_recognizer = LazyTesseractRecognizer()
ocr_assistant = OCRAssistant()
