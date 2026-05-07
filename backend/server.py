import asyncio
import json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

pcs = set()
relay = MediaRelay()
active_video_track = None

routes = web.RouteTableDef()

# -----------------------------
# BASIC ROUTES
# -----------------------------
@routes.get("/")
async def index(request):
    return web.Response(text="WebRTC Server Running")

# -----------------------------
# OFFER (CORE)
# -----------------------------
@routes.post("/offer")
async def offer(request):
    global active_video_track

    params = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        global active_video_track
        print("Track received:", track.kind)

        if track.kind == "video":
            active_video_track = track

    # Set remote (client offer)
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    )

    # If viewer, send video back
    if active_video_track:
        print("Sending video to viewer")
        pc.addTrack(relay.subscribe(active_video_track))

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })

# -----------------------------
# CLEANUP
# -----------------------------
async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

# -----------------------------
# APP START
# -----------------------------
app = web.Application()
app.add_routes(routes)
app.on_shutdown.append(on_shutdown)

web.run_app(app, host="0.0.0.0", port=8080)