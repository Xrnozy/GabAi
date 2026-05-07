import asyncio
import json
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from color_detection import closest_color_name, color_assistant


def resize_preserve_aspect(frame, max_dim=320):
    h, w = frame.shape[:2]
    if max(h, w) == 0:
        return frame
    scale = max_dim / float(max(h, w))
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def _center_roi(frame_bgr: np.ndarray, half_size: int = 50) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    x1 = int(np.clip(cx - half_size, 0, w - 1))
    y1 = int(np.clip(cy - half_size, 0, h - 1))
    x2 = int(np.clip(cx + half_size, x1 + 1, w))
    y2 = int(np.clip(cy + half_size, y1 + 1, h))
    roi = frame_bgr[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


class ColorVideoTrack(VideoStreamTrack):
    """
    Color-only inference track.
    Detects the closest named color from a center ROI and emits it over the navigation data channel.
    """

    def __init__(self, track, pcs_ref: set, infer_every_s: float = 0.15, roi_half_size: int = 50):
        super().__init__()
        self.track = track
        self.pcs_ref = pcs_ref
        self.infer_every_s = infer_every_s
        self.roi_half_size = roi_half_size

        self._last_infer_time = 0.0
        self._last_name: str = ""
        self._last_rgb: Tuple[int, int, int] = (0, 0, 0)

    async def recv(self):
        frame = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_small = resize_preserve_aspect(frame_bgr, 480)

        now = time.time()
        do_infer = (now - self._last_infer_time) >= self.infer_every_s

        name: Optional[str] = None
        conf: float = 0.0
        should_speak: bool = False
        rgb: Tuple[int, int, int] = (0, 0, 0)

        if do_infer:
            self._last_infer_time = now

            def _infer_local(buf_bgr: np.ndarray):
                roi, _ = _center_roi(buf_bgr, self.roi_half_size)
                if roi.size == 0:
                    return None, (0, 0, 0), 0.0
                avg_bgr = roi.mean(axis=(0, 1))
                avg_rgb = tuple(int(x) for x in avg_bgr[::-1])
                nm, c = closest_color_name(avg_rgb)
                return nm, avg_rgb, float(c)

            name, rgb, conf = await asyncio.to_thread(_infer_local, frame_small)
            decision = color_assistant.decide(name=name, rgb=rgb, confidence=conf)
            self._last_name = decision.name
            self._last_rgb = decision.rgb
            should_speak = decision.should_speak

            nav_data = {
                "nav_command": decision.name or "unknown",  # keep mobile listener compatible
                "confidence": round(decision.confidence, 3),
                "should_speak": bool(should_speak),
                "rgb": list(decision.rgb),
                "mode": "color",
            }
            try:
                payload = json.dumps(nav_data)
                for pc in list(self.pcs_ref):
                    if pc.connectionState != "connected":
                        continue
                    if getattr(pc, "_processing_mode", "navigation") != "color":
                        continue
                    ch = getattr(pc, "navigation_channel", None)
                    if ch and ch.readyState == "open":
                        try:
                            ch.send(payload)
                        except Exception:
                            pass
            except Exception:
                pass

        # Overlay ROI + label for preview/debug.
        overlay = frame_small
        try:
            roi, (x1, y1, x2, y2) = _center_roi(overlay, self.roi_half_size)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            shown = self._last_name or "..."
            cv2.putText(
                overlay,
                f"COLOR: {shown}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"RGB: {self._last_rgb}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

        out = VideoFrame.from_ndarray(overlay, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out

