# ---------------------------------------------------------------------------
# video_tracks.py – WebRTC VideoStreamTrack subclasses
# ---------------------------------------------------------------------------
"""
Each track runs inference EVERY FRAME (no frame-skipping) for temporal
consistency and smooth directional decision-making.
"""
import asyncio
import time
import json
from typing import Optional

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

def resize_preserve_aspect(frame, max_dim=320):
    h, w = frame.shape[:2]
    if max(h, w) == 0:
        return frame
    scale = max_dim / float(max(h, w))
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

from config import (
    YOLO_INPUT_SIZE, shared_nav_state, DirectionalRiskResult,
    ROAD_PATH_INPUT_SIZE, ROAD_PATH_CONFIDENCE,
)
from depth import DepthCache, analyse_depth_regions, run_depth_map, infer_depth_f32
from detection import run_yolo_inference, run_pothole_inference, run_yolo_raw
from optical_flow import (
    compute_flow_vectors, compute_avg_flow_magnitude,
    run_optical_flow_inference,
)
from road_seg import run_road_path_inference
from directional_safety import safety_field
from navigation import (
    compute_unified_detection, send_navigation_data, nav_assistant,
)
from visualization import (
    draw_full_directional_overlay, draw_depth_regions,
)


# ===========================================================================
# YOLO detection track
# ===========================================================================

class YoloVideoTrack(VideoStreamTrack):
    def __init__(self, track, pcs=None):
        super().__init__()
        self.track = track
        self.pcs = pcs if pcs else set()
        self.last_time = time.time()

    async def recv(self):
        frame     = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = resize_preserve_aspect(frame_bgr, 320)
        
        result = await asyncio.to_thread(run_yolo_raw, frame_bgr)
        annotated = result.plot() if result else frame_bgr
        
        if annotated.shape[:2] != frame_bgr.shape[:2]:
            annotated = cv2.resize(annotated, (frame_bgr.shape[1], frame_bgr.shape[0]))
            
        now = time.time()
        fps = 1.0 / (now - self.last_time) if now - self.last_time > 0 else 0
        self.last_time = now

        # Send stats via data channel
        if self.pcs:
            detections = []
            if result and result.boxes:
                for box in result.boxes:
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    name = result.names[cls_id] if isinstance(result.names, dict) else str(cls_id)
                    detections.append({
                        "class": name,
                        "confidence": conf
                    })
            stats_json = json.dumps({
                "type": "detection_stats",
                "detections": detections,
                "fps": round(fps, 1)
            })
            for pc in list(self.pcs):
                if pc.connectionState != "connected":
                    continue
                ch = getattr(pc, 'detection_channel', None)
                if ch and ch.readyState == "open":
                    try: ch.send(stats_json)
                    except: pass

        out = VideoFrame.from_ndarray(annotated, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out


# ===========================================================================
# Depth estimation track
# ===========================================================================

class DepthVideoTrack(VideoStreamTrack):
    def __init__(self, track, pcs=None):
        super().__init__()
        self.track        = track
        self._depth_cache = DepthCache()
        self.pcs          = pcs if pcs else set()
        self.last_time    = time.time()

    async def recv(self):
        frame     = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = resize_preserve_aspect(frame_bgr, 320)
        await asyncio.to_thread(self._depth_cache.update, frame_bgr)
        
        min_depth = 0.0
        max_depth = 0.0
        
        if self._depth_cache.depth_u8 is not None:
            coloured = cv2.applyColorMap(
                self._depth_cache.depth_u8, cv2.COLORMAP_INFERNO)
            if self._depth_cache.depth_f32 is not None:
                dr       = analyse_depth_regions(self._depth_cache.depth_f32)
                coloured = draw_depth_regions(coloured, dr)
                min_depth = float(np.min(self._depth_cache.depth_f32))
                max_depth = float(np.max(self._depth_cache.depth_f32))
        else:
            coloured = frame_bgr
            
        if coloured.shape[:2] != frame_bgr.shape[:2]:
            coloured = cv2.resize(coloured, (frame_bgr.shape[1], frame_bgr.shape[0]))
            
        now = time.time()
        fps = 1.0 / (now - self.last_time) if now - self.last_time > 0 else 0
        self.last_time = now

        # Send stats via data channel
        if self.pcs:
            stats_json = json.dumps({
                "type": "depth_stats",
                "min_depth": round(min_depth, 2),
                "max_depth": round(max_depth, 2),
                "fps": round(fps, 1)
            })
            for pc in list(self.pcs):
                if pc.connectionState != "connected":
                    continue
                ch = getattr(pc, 'depth_channel', None)
                if ch and ch.readyState == "open":
                    try: ch.send(stats_json)
                    except: pass

        out = VideoFrame.from_ndarray(coloured, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out


# ===========================================================================
# Optical flow track
# ===========================================================================

class OpticalFlowVideoTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track     = track
        self.prev_gray = None

    async def recv(self):
        frame     = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = resize_preserve_aspect(frame_bgr, 320)
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            flow_frame = frame_bgr
        else:
            flow_frame = await asyncio.to_thread(
                run_optical_flow_inference,
                self.prev_gray, curr_gray, frame_bgr)
            self.prev_gray = curr_gray
        out = VideoFrame.from_ndarray(flow_frame, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out


# ===========================================================================
# Directional Safety Fusion track (PRIMARY intelligence track)
# ===========================================================================

def _run_directional_fusion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    frame_bgr: np.ndarray,
    segmentation_mask: Optional[np.ndarray],
    pothole_detected: bool,
    pothole_area_ratio: float,
) -> tuple:
    """
    Combined pipeline: YOLO + depth + optical flow + directional safety field.
    Runs every frame for temporal consistency.
    
    Returns: (annotated_frame, dir_result, avg_flow_mag)
    """
    height, width = frame_bgr.shape[:2]

    # 1. YOLO detection
    yolo_result = run_yolo_raw(frame_bgr)
    boxes = yolo_result.boxes if yolo_result else None

    # 2. Depth estimation
    depth_f32    = infer_depth_f32(frame_bgr, height, width)
    depth_map_u8 = (depth_f32 * 255.0).astype(np.uint8)

    # 3. Optical flow
    flow_vectors = compute_flow_vectors(prev_gray, curr_gray)
    avg_flow_mag = compute_avg_flow_magnitude(flow_vectors)

    # 4. Directional Safety Field computation
    dir_result = safety_field.compute(
        frame_bgr=frame_bgr,
        depth_f32=depth_f32,
        depth_map_u8=depth_map_u8,
        flow_vectors=flow_vectors,
        avg_flow_mag=avg_flow_mag,
        boxes=boxes,
        result=yolo_result,
        segmentation_mask=segmentation_mask,
        pothole_detected=pothole_detected,
        pothole_area_ratio=pothole_area_ratio,
    )

    # 5. Visualize
    fan_regions = safety_field.get_fan_regions()
    overlay = draw_full_directional_overlay(
        frame_bgr, dir_result, fan_regions,
        flow_vectors, avg_flow_mag)

    return overlay, dir_result, avg_flow_mag


class DirectionalFusionVideoTrack(VideoStreamTrack):
    """
    PRIMARY intelligence track: replaces DistanceFusionVideoTrack.
    Uses the new Directional Safety Field system.
    Runs all modules every frame for temporal consistency.
    """
    def __init__(self, track, pcs_ref: set):
        super().__init__()
        self.track     = track
        self.prev_gray = None
        self._pcs      = pcs_ref

    async def recv(self):
        frame     = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = resize_preserve_aspect(frame_bgr, 320)
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            overlay = frame_bgr.copy()
            cv2.putText(overlay, "Directional Safety Field warming up...",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            seg_mask = shared_nav_state.segmentation_mask

            overlay, dir_result, avg_flow_mag = await asyncio.to_thread(
                _run_directional_fusion,
                self.prev_gray, curr_gray, frame_bgr, seg_mask,
                shared_nav_state.pothole_detected, shared_nav_state.pothole_area_ratio)

            # Update shared state
            shared_nav_state.directional_result = dir_result

            # Build unified detection and emit navigation
            unified_result = compute_unified_detection(
                dir_result=dir_result,
                pothole_detected=shared_nav_state.pothole_detected,
                pothole_area_ratio=shared_nav_state.pothole_area_ratio,
                road_direction=shared_nav_state.road_direction,
                walkable_pct=shared_nav_state.walkable_pct,
            )
            shared_nav_state.unified_detection = unified_result
            instruction = nav_assistant.emit_unified(unified_result)
            should_speak = instruction is not None
            send_navigation_data(unified_result, self._pcs, should_speak)

            self.prev_gray = curr_gray

        out = VideoFrame.from_ndarray(overlay, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out


# ===========================================================================
# Pothole detection track
# ===========================================================================

class PotholeVideoTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame     = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = resize_preserve_aspect(frame_bgr, 320)
        pothole_frame, detected, area_ratio = await asyncio.to_thread(
            run_pothole_inference, frame_bgr)
        shared_nav_state.pothole_detected   = detected
        shared_nav_state.pothole_area_ratio  = area_ratio
        if pothole_frame.shape[:2] != frame_bgr.shape[:2]:
            pothole_frame = cv2.resize(pothole_frame,
                                       (frame_bgr.shape[1], frame_bgr.shape[0]))
        out = VideoFrame.from_ndarray(pothole_frame, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out


# ===========================================================================
# Road-path segmentation track
# ===========================================================================

class RoadPathVideoTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame     = await self.track.recv()
        frame_bgr = frame.to_ndarray(format="bgr24")
        frame_bgr = resize_preserve_aspect(frame_bgr, 320)

        path_frame, direction, walkable_pct, walkable_mask = await asyncio.to_thread(
            run_road_path_inference, frame_bgr)
        shared_nav_state.road_direction = direction
        shared_nav_state.walkable_pct   = walkable_pct

        # Extract walkable mask for directional safety field
        if walkable_mask is not None:
            shared_nav_state.segmentation_mask = walkable_mask

        if path_frame.shape[:2] != frame_bgr.shape[:2]:
            path_frame = cv2.resize(path_frame,
                                    (frame_bgr.shape[1], frame_bgr.shape[0]))
        out = VideoFrame.from_ndarray(path_frame, format="bgr24")
        out.pts, out.time_base = frame.pts, frame.time_base
        return out
