import asyncio
import json
import time
from typing import Optional

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from asl_detection import asl_assistant, asl_recognizer, ASLAssistant


def resize_preserve_aspect(frame, max_dim=320):
    h, w = frame.shape[:2]
    if max(h, w) == 0:
        return frame
    scale = max_dim / float(max(h, w))
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


class AslVideoTrack(VideoStreamTrack):
    """
    ASL-only inference track.
    Runs only the ASL YOLO detector and emits recognized words to the mobile data channel.
    """

    def __init__(self, track, pcs_ref: set, speak_every_s: float = 0.25):
        super().__init__()
        self.track = track
        self.pcs_ref = pcs_ref
        self.speak_every_s = speak_every_s

        self._last_infer_time = 0.0
        self._last_label: str = ""

    async def recv(self):
        frame = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = resize_preserve_aspect(frame_bgr, 320)

        now = time.time()
        do_infer = (now - self._last_infer_time) >= self.speak_every_s

        label: Optional[str] = None
        conf: float = 0.0
        should_speak: bool = False

        if do_infer:
            self._last_infer_time = now
            label, conf = await asyncio.to_thread(asl_recognizer.infer, frame_bgr)
            decision = asl_assistant.decide(label=label, confidence=conf)
            label = decision.nav_command
            should_speak = decision.should_speak
            self._last_label = label

            # Send to all peers with an OPEN navigation data channel.
            nav_data = {
                "nav_command": label,
                "confidence": round(conf, 3),
                "should_speak": bool(should_speak),
            }
            try:
                payload = json.dumps(nav_data)
                for pc in list(self.pcs_ref):
                    if pc.connectionState != "connected":
                        continue
                    # Only deliver ASL words to ASL-mode peers.
                    if getattr(pc, "_processing_mode", "navigation") != "asl":
                        continue
                    ch = getattr(pc, "navigation_channel", None)
                    if ch and ch.readyState == "open":
                        try:
                            ch.send(payload)
                        except Exception:
                            pass
            except Exception:
                pass

        # Minimal overlay so local debugging is easier (no heavy drawing each frame).
        overlay = frame_bgr
        if self._last_label:
            try:
                cv2.putText(
                    overlay,
                    f"ASL: {self._last_label}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

        out = VideoFrame.from_ndarray(overlay, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out

