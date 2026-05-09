import asyncio
import json
import time
from typing import Optional

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from ocr_detection import ocr_recognizer, ocr_assistant


def resize_preserve_aspect(frame, max_dim=320):
    h, w = frame.shape[:2]
    if max(h, w) == 0:
        return frame
    scale = max_dim / float(max(h, w))
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def _preprocess_for_ocr(frame_bgr):
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    except Exception:
        return frame_bgr


class OcrVideoTrack(VideoStreamTrack):
    """
    Tesseract-only inference track.
    Runs pytesseract and emits recognized text to the mobile navigation data channel.
    """

    def __init__(self, track, pcs_ref: set, infer_every_s: float = 0.25, ocr_max_dim: int = 960):
        super().__init__()
        self.track = track
        self.pcs_ref = pcs_ref
        self.infer_every_s = infer_every_s
        self.ocr_max_dim = ocr_max_dim
        self._last_infer_time = 0.0
        self._last_text: str = ""
        self._inference_in_flight: bool = False

    async def recv(self):
        frame = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")

        frame_for_ocr = resize_preserve_aspect(frame_bgr, self.ocr_max_dim)
        frame_for_ocr = _preprocess_for_ocr(frame_for_ocr)

        now = time.time()
        do_infer = (now - self._last_infer_time) >= self.infer_every_s

        text: Optional[str] = None
        conf: float = 0.0
        should_speak: bool = False

        if do_infer:
            self._last_infer_time = now
            if not self._inference_in_flight:
                self._inference_in_flight = True
                try:
                    text, conf = await asyncio.to_thread(ocr_recognizer.infer, frame_for_ocr)
                    decision = ocr_assistant.decide(text=text, confidence=conf)
                    text = decision.nav_command
                    should_speak = decision.should_speak
                    self._last_text = text
                finally:
                    self._inference_in_flight = False

            nav_data = {
                "nav_command": text,
                "confidence": round(conf, 3),
                "should_speak": bool(should_speak),
                "mode": "ocr",
            }
            try:
                payload = json.dumps(nav_data)
                for pc in list(self.pcs_ref):
                    if pc.connectionState != "connected":
                        continue
                    if getattr(pc, "_processing_mode", "navigation") != "ocr":
                        continue
                    ch = getattr(pc, "navigation_channel", None)
                    if ch and ch.readyState == "open":
                        try:
                            ch.send(payload)
                        except Exception:
                            pass
            except Exception:
                pass

        overlay = resize_preserve_aspect(frame_bgr, 640)
        if self._last_text:
            try:
                shown = self._last_text
                if len(shown) > 42:
                    shown = shown[:42] + "..."
                cv2.putText(
                    overlay,
                    f"OCR: {shown}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

        out = VideoFrame.from_ndarray(overlay, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out
