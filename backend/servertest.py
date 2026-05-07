# ===========================================================================
# servertest.py – Slim server: routes, WebRTC signalling, app bootstrap
# ===========================================================================
"""
All inference logic, navigation systems, and visualization have been
modularized into:
    config.py             – constants, model loading, shared state
    depth.py              – depth estimation pipeline
    detection.py          – YOLO + pothole + label utilities
    optical_flow.py       – optical flow tracking
    road_seg.py           – road/path segmentation
    directional_safety.py – NEW: directional safety field (core nav)
    navigation.py         – navigation assistant + unified detection
    visualization.py      – all overlay drawing
    video_tracks.py       – WebRTC VideoStreamTrack subclasses
"""
import asyncio
import json

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

# ── Module imports ─────────────────────────────────────────────────────────
from config import shared_nav_state, DirectionalRiskResult
from navigation import nav_assistant
from video_tracks import (
    YoloVideoTrack,
    DepthVideoTrack,
    OpticalFlowVideoTrack,
    DirectionalFusionVideoTrack,
    PotholeVideoTrack,
    RoadPathVideoTrack,
)

# ASL-only pipeline (no navigation/depth/optical-flow)
from asl_video_tracks import AslVideoTrack
from ocr_video_tracks import OcrVideoTrack
from color_video_tracks import ColorVideoTrack


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pcs: set = set()
relay = MediaRelay()
active_video_track = None
routes = web.RouteTableDef()

# Source generation counter: incremented every time a new sender connects.
# Lets viewers know they're watching a stale source.
source_generation: int = 0

# Set of SSE response queues – each connected viewer's /events endpoint
# gets a queue. When source changes, we push an event into each queue.
sse_queues: set = set()

# Track which PCs are viewers vs senders
viewer_pcs: set = set()
sender_pc = None
active_processing_mode: str = "navigation"  # "navigation" or "asl" or "ocr" or "color"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def subscribe_relay_track(track):
    try:    return relay.subscribe(track, buffered=False)
    except: return relay.subscribe(track)


async def notify_viewers_source_changed():
    """
    Push a source_ready event to all SSE listeners immediately.
    This makes viewers reconnect the instant the Android app connects.
    """
    event_data = json.dumps({
        "event": "source_ready",
        "generation": source_generation,
    })
    dead_queues = set()
    for q in list(sse_queues):
        try:
            q.put_nowait(event_data)
        except Exception:
            dead_queues.add(q)
    sse_queues.difference_update(dead_queues)


async def close_stale_viewer_pcs():
    """
    Close all existing viewer PCs so they detect disconnection
    and auto-reconnect with fresh tracks from the new source.
    """
    stale = list(viewer_pcs)
    if not stale:
        return
    print(f"[SOURCE] Closing {len(stale)} stale viewer connection(s) for re-negotiation")
    for vpc in stale:
        viewer_pcs.discard(vpc)
        pcs.discard(vpc)
        try:
            await vpc.close()
        except Exception:
            pass


# ===========================================================================
# SDP validation helper
# ===========================================================================

def validate_sdp(sdp_string: str, label: str = "SDP") -> bool:
    print(f"\nValidating {label}:")
    lines = sdp_string.split("\n")
    mids  = []
    bundle_group = None
    for line in lines:
        if line.startswith("m="):    print(f"  Media section: {line[:50]}")
        if line.startswith("a=mid:"):
            mid = line.split("a=mid:")[1].strip()
            mids.append(mid)
            print(f"  Found MID: '{mid}'")
        if line.startswith("a=group:BUNDLE"):
            bundle_group = line.split("a=group:BUNDLE")[1].strip()
            print(f"  BUNDLE group: '{bundle_group}'")
    if bundle_group:
        for bmid in bundle_group.split():
            if not bmid:
                print("  ERROR: Empty MID in BUNDLE group!")
                return False
            if bmid not in mids:
                print(f"  ERROR: BUNDLE references MID '{bmid}' not in SDP!")
                return False
    print(f"  SDP validation passed (MIDs: {mids})\n")
    return len(mids) > 0


# ===========================================================================
# HTTP routes
# ===========================================================================

@routes.get("/")
async def index(request):
    return web.FileResponse("../frontend/src/index.html")


@routes.get("/stream-status")
async def stream_status(request):
    source_ready = (
        active_video_track is not None
        and getattr(active_video_track, "readyState", "live") == "live"
    )
    return web.json_response({
        "source_ready": source_ready,
        "generation":   source_generation,
    })


@routes.get("/events")
async def sse_events(request):
    """
    Server-Sent Events endpoint.
    Pushes instant notifications when the video source changes
    (Android app connects/disconnects), so the viewer auto-reconnects
    without needing a manual page refresh.
    """
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type":  "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    queue = asyncio.Queue()
    sse_queues.add(queue)
    print("[SSE] Viewer subscribed to source events")

    # Send initial status immediately so the viewer knows current state
    initial = json.dumps({
        "event": "status",
        "source_ready": active_video_track is not None
                        and getattr(active_video_track, "readyState", "live") == "live",
        "generation": source_generation,
    })
    await response.write(f"data: {initial}\n\n".encode())

    try:
        while True:
            try:
                # Wait for an event (with periodic keepalive)
                data = await asyncio.wait_for(queue.get(), timeout=15.0)
                await response.write(f"data: {data}\n\n".encode())
            except (asyncio.TimeoutError, TimeoutError):
                # Send keepalive comment to prevent connection timeout
                try:
                    await response.write(b": keepalive\n\n")
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    break
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                break
    except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError):
        pass
    finally:
        sse_queues.discard(queue)
        print("[SSE] Viewer unsubscribed from source events")

    return response


@routes.get("/nav-state")
async def nav_state(request):
    dr = shared_nav_state.directional_result
    dr_payload = None
    if dr is not None:
        dr_payload = {
            "direction_risks":  {k: round(v, 3) for k, v in dr.direction_risks.items()},
            "recommended_dir":  dr.recommended_dir,
            "nav_command":      dr.nav_command,
            "nav_angle_deg":    round(dr.nav_angle_deg, 1),
            "is_emergency":     dr.is_emergency_stop,
            "scene_risk":       round(dr.scene_risk, 3),
            "confidence":       round(dr.confidence, 3),
            "direction_free_space": {k: round(v, 2) for k, v in dr.direction_free_space.items()},
            "top_risk_object": (
                {
                    "label":      dr.object_risks[0]["label"],
                    "risk_score": round(dr.object_risks[0]["risk_score"], 3),
                    "direction":  dr.object_risks[0].get("direction", "forward"),
                }
                if dr.object_risks else None
            ),
            "safest_direction":      dr.safest_direction,
            "steering_strength":     round(dr.steering_strength, 2),
            "lane_confidence":       round(dr.lane_confidence, 3),
            "collision_probability": round(dr.collision_probability, 3),
            "path_continuity_score": round(dr.path_continuity_score, 3),
            "future_risk_score":     round(dr.future_risk_score, 3),
        }
    return web.json_response({
        "road_direction":    shared_nav_state.road_direction,
        "walkable_pct":      round(shared_nav_state.walkable_pct, 1),
        "pothole_detected":  shared_nav_state.pothole_detected,
        "last_instruction":  nav_assistant._last_instruction,
        "directional_field": dr_payload,
    })


@routes.post("/offer")
async def offer(request):
    global active_video_track, source_generation, sender_pc
    try:
        params    = await request.json()
        req_mode = str(params.get("mode", "navigation")).strip().lower()
        if req_mode not in ("navigation", "asl", "ocr", "color"):
            req_mode = "navigation"
        sdp_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        validate_sdp(params["sdp"], "Client Offer")
        pc = RTCPeerConnection()
        pcs.add(pc)
        pc._processing_mode = req_mode
        pc.addTransceiver("video", direction="sendrecv")
        track_event = asyncio.Event()

        # Create negotiated data channels server-side (must match client's negotiated IDs)
        det_ch = pc.createDataChannel("detection", negotiated=True, id=0)
        dep_ch = pc.createDataChannel("depth",     negotiated=True, id=1)
        nav_ch = pc.createDataChannel("navigation", negotiated=True, id=2)
        pc.detection_channel  = det_ch
        pc.depth_channel      = dep_ch
        pc.navigation_channel = nav_ch
        print("[DC] Created negotiated data channels (detection=0, depth=1, navigation=2)")

        @det_ch.on("open")
        def _det_open():
            print("[DC] Detection channel OPEN")

        @dep_ch.on("open")
        def _dep_open():
            print("[DC] Depth channel OPEN")

        @nav_ch.on("open")
        def _nav_open():
            print("[DC] Navigation channel OPEN")

        @pc.on("track")
        def on_track(track):
            global active_video_track, source_generation, sender_pc
            print(f"Track received: {track.kind}")
            if track.kind == "video":
                active_video_track = track
                source_generation += 1
                sender_pc = pc
                global active_processing_mode
                active_processing_mode = getattr(pc, "_processing_mode", "navigation")
                print(f"[MODE] Active processing mode set to: {active_processing_mode}")
                print(f"[SOURCE] New video source registered (generation {source_generation})")
                track_event.set()

                # Close stale viewers & notify SSE listeners immediately
                asyncio.ensure_future(_on_new_source())

        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            state = pc.connectionState
            print(f"[PC] Connection state: {state}")
            if state in ("failed", "closed", "disconnected"):
                # If this was the sender, notify viewers source is gone
                if pc is sender_pc:
                    print("[SOURCE] Sender disconnected")
                    await notify_viewers_source_changed()
                # Clean up
                pcs.discard(pc)
                viewer_pcs.discard(pc)

        await pc.setRemoteDescription(sdp_offer)

        sender_track_received = False
        try:
            await asyncio.wait_for(track_event.wait(), timeout=2.0)
            sender_track_received = True
        except asyncio.TimeoutError:
            print("No track received within timeout (receive-only client)")

        if not sender_track_received and active_video_track:
            print("Viewer connected: adding all 7 processed tracks")
            if active_processing_mode == "asl":
                raw_track = subscribe_relay_track(active_video_track)
                pc.addTrack(AslVideoTrack(raw_track, pcs))
                print("ASL viewer: added ASL-only track (no navigation/depth/flow)")
            elif active_processing_mode == "ocr":
                # Provide a raw preview track (slot 0) + OCR overlay track (slot 1),
                # matching the navigation viewer layout (live feed + analysis track).
                raw_preview = subscribe_relay_track(active_video_track)
                ocr_input = subscribe_relay_track(active_video_track)
                pc.addTrack(raw_preview)
                pc.addTrack(OcrVideoTrack(ocr_input, pcs))
                print("OCR viewer: added raw preview + OCR overlay track")
            elif active_processing_mode == "color":
                raw_preview = subscribe_relay_track(active_video_track)
                color_input = subscribe_relay_track(active_video_track)
                pc.addTrack(raw_preview)
                pc.addTrack(ColorVideoTrack(color_input, pcs))
                print("COLOR viewer: added raw preview + color overlay track")
            else:
                raw_track       = subscribe_relay_track(active_video_track)
                yolo_input      = subscribe_relay_track(active_video_track)
                depth_input     = subscribe_relay_track(active_video_track)
                flow_input      = subscribe_relay_track(active_video_track)
                distance_input  = subscribe_relay_track(active_video_track)
                pothole_input   = subscribe_relay_track(active_video_track)
                road_path_input = subscribe_relay_track(active_video_track)

                pc.addTrack(raw_track)
                pc.addTrack(YoloVideoTrack(yolo_input, pcs))
                pc.addTrack(DepthVideoTrack(depth_input, pcs))
                pc.addTrack(OpticalFlowVideoTrack(flow_input))
                pc.addTrack(DirectionalFusionVideoTrack(distance_input, pcs))
                pc.addTrack(PotholeVideoTrack(pothole_input))
                pc.addTrack(RoadPathVideoTrack(road_path_input))
                print("All 7 tracks added to viewer peer connection")

            # Mark this PC as a viewer
            viewer_pcs.add(pc)
            # Tag the generation so we know if it becomes stale
            pc._source_generation = source_generation

        elif sender_track_received:
            print("Sender connected: video source registered")
            if req_mode == "asl" and active_video_track:
                raw_track = subscribe_relay_track(active_video_track)
                pc.addTrack(AslVideoTrack(raw_track, pcs))
                print("ASL sender: added ASL-only track (no navigation/depth/flow)")
            elif req_mode == "ocr" and active_video_track:
                raw_track = subscribe_relay_track(active_video_track)
                pc.addTrack(OcrVideoTrack(raw_track, pcs))
                print("OCR sender: added OCR-only track (text recognition mode)")
            elif req_mode == "color" and active_video_track:
                raw_track = subscribe_relay_track(active_video_track)
                pc.addTrack(ColorVideoTrack(raw_track, pcs))
                print("COLOR sender: added color-only track (color understanding mode)")
        else:
            print("Viewer connected but no active video source yet")
            viewer_pcs.add(pc)
            pc._source_generation = -1  # no source at connect time

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        validate_sdp(pc.localDescription.sdp, "Answer")

        return web.json_response({
            "sdp":  pc.localDescription.sdp,
            "type": pc.localDescription.type,
        })

    except Exception as exc:
        print(f"Error in offer handler: {exc}")
        import traceback; traceback.print_exc()
        return web.json_response({"error": str(exc)}, status=500)


async def _on_new_source():
    """Called when a new video source (Android app) connects."""
    # 1. Close stale viewer connections so they auto-reconnect with tracks
    await close_stale_viewer_pcs()
    # 2. Notify SSE listeners to reconnect instantly
    await notify_viewers_source_changed()


# ===========================================================================
# Shutdown & app bootstrap
# ===========================================================================

async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()
    viewer_pcs.clear()


app = web.Application()
app.add_routes(routes)
app.on_shutdown.append(on_shutdown)

web.run_app(app, host="0.0.0.0", port=8080)