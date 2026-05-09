import time
from dataclasses import dataclass
from typing import Optional, Tuple
import pytesseract

import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


class LazyTesseractRecognizer:
    """
    Tesseract recognizer for real-time OCR mode.
    Returns (text, confidence) where confidence is [0, 1].
    """

    def __init__(
        self,
        max_dim: int = 800,
        min_word_conf: float = 0.45,
        min_text_conf: float = 0.35,
        min_word_chars: int = 2,
    ):
        self._backend: Optional[str] = None
        self._pytesseract = None
        self._max_dim = max_dim
        self._min_word_conf = min_word_conf
        self._min_text_conf = min_text_conf
        self._min_word_chars = min_word_chars
        self._lang = "eng"
        self._config = "--oem 1 --psm 6"

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

    def _resize_for_speed(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._max_dim <= 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if max(h, w) <= self._max_dim:
            return frame_bgr
        scale = self._max_dim / float(max(h, w))
        return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

    def _filter_token(self, text: str, conf: float) -> Optional[str]:
        t = _clean_text(text)
        if not t:
            return None
        alnum = sum(ch.isalnum() for ch in t)
        if alnum == 0:
            return None
        if len(t) < self._min_word_chars and not t.isdigit():
            return None
        if len(t) >= 3 and (alnum / len(t)) < 0.5:
            return None
        if conf < self._min_word_conf:
            return None
        return t

    def infer(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        self._ensure_loaded()
        if self._backend != "pytesseract" or self._pytesseract is None:
            return None, 0.0

        try:
            frame_bgr = self._resize_for_speed(frame_bgr)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            data = self._pytesseract.image_to_data(
                rgb,
                output_type=self._pytesseract.Output.DICT,
                lang=self._lang,
                config=self._config,
            )

            texts = []
            confs = []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                try:
                    c_raw = float(conf)
                except Exception:
                    continue
                if c_raw < 0.0:
                    continue
                c = c_raw / 100.0
                t = self._filter_token(str(text), c)
                if not t:
                    continue
                texts.append(t)
                confs.append(c)

            merged = _clean_text(" ".join(texts))
            if not merged:
                return None, 0.0
            confidence = float(np.clip(np.mean(confs), 0.0, 1.0)) if confs else 0.0
            if confidence < self._min_text_conf:
                return None, 0.0
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
