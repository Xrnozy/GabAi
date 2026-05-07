#!/usr/bin/env python3
"""
windows_sender.py  — runs on Windows (plain Python, no ROS)
════════════════════════════════════════════════════════════════
1. Auto-detects WSL IP at runtime (no hardcoding needed)
2. Captures webcam frames with OpenCV
3. JPEG-encodes and streams them to WSL (wsl_bridge.py)
4. Receives packed point cloud data back from WSL
5. Visualizes the live point cloud with Open3D

Install deps (Windows cmd / PowerShell):
  pip install opencv-python numpy open3d

Run:
  python windows_sender.py                        # fully automatic
  python windows_sender.py --wsl-ip 192.168.1.5  # manual override
  python windows_sender.py --camera 1 --fps 15   # different camera/fps
════════════════════════════════════════════════════════════════
"""

import argparse
import socket
import struct
import subprocess
import threading
import time
import sys
from collections import deque

import cv2
import numpy as np

try:
    import open3d as o3d
    OPEN3D = True
except ImportError:
    OPEN3D = False
    print("[warn] open3d not found — point cloud window disabled.")
    print("       pip install open3d")


# ── Config ────────────────────────────────────────────────────
VIDEO_PORT   = 9000
CLOUD_PORT   = 9001
JPEG_QUALITY = 75    # 0-100; lower = faster but blurrier


# ═══════════════════════════════════════════════════════════════
# WSL IP auto-detection
# ═══════════════════════════════════════════════════════════════

def get_wsl_ip() -> str:
    print("[auto] Detecting WSL IP...")

    # Method 1: get eth0 specifically (avoids Docker bridge 172.17.x.x)
    try:
        result = subprocess.run(
            ["wsl", "ip", "addr", "show", "eth0"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                ip = line.split()[1].split("/")[0]
                print(f"[auto] WSL eth0 IP: {ip}")
                return ip
    except Exception as e:
        print(f"[auto] Method 1 failed: {e}")

    # Method 2: hostname -I but skip Docker IPs (172.17.x.x)
    try:
        result = subprocess.run(
            ["wsl", "hostname", "-I"],
            capture_output=True, text=True, timeout=5
        )
        for ip in result.stdout.strip().split():
            if not ip.startswith("172.17."):   # skip Docker bridge
                print(f"[auto] WSL IP: {ip}")
                return ip
    except Exception as e:
        print(f"[auto] Method 2 failed: {e}")

    # Method 3: resolv.conf nameserver (Windows host IP — works in reverse)
    try:
        result = subprocess.run(
            ["wsl", "cat", "/etc/resolv.conf"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("nameserver"):
                ip = line.split()[1].strip()
                print(f"[auto] WSL IP from resolv.conf: {ip}")
                return ip
    except Exception as e:
        print(f"[auto] Method 3 failed: {e}")

    print("[auto] Falling back to 127.0.0.1")
    return "127.0.0.1"

# ═══════════════════════════════════════════════════════════════
# TCP helpers  (length-prefixed: 4-byte LE uint32 + payload)
# ═══════════════════════════════════════════════════════════════

def recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> bytes | None:
    header = recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack("<I", header)
    return recv_exact(sock, length)


def send_msg(sock: socket.socket, data: bytes) -> bool:
    try:
        sock.sendall(struct.pack("<I", len(data)) + data)
        return True
    except OSError:
        return False


# ═══════════════════════════════════════════════════════════════
# Unpack point cloud wire format from WSL
# Wire format: [uint32 N] [N × (float32 x, y, z  +  uint8 r, g, b)]
# ═══════════════════════════════════════════════════════════════

def unpack_cloud(data: bytes):
    """Returns (xyz: N×3 float32,  rgb: N×3 float32 0-1) or (None, None)."""
    if len(data) < 4:
        return None, None

    (n,) = struct.unpack("<I", data[:4])
    if n == 0:
        return None, None

    stride = 4 * 3 + 3          # 15 bytes per point (xyz float32 + rgb uint8)
    expected = 4 + n * stride
    if len(data) < expected:
        return None, None

    raw = np.frombuffer(data[4:4 + n * stride], dtype=np.uint8).reshape(n, stride)

    xyz = np.frombuffer(raw[:, :12].tobytes(), dtype=np.float32).reshape(n, 3)
    r = raw[:, 12].astype(np.float32) / 255.0
    g = raw[:, 13].astype(np.float32) / 255.0
    b = raw[:, 14].astype(np.float32) / 255.0
    rgb = np.stack([r, g, b], axis=1)

    return xyz, rgb


# ═══════════════════════════════════════════════════════════════
# Video sender thread
# ═══════════════════════════════════════════════════════════════

def video_sender(wsl_ip: str, camera_index: int, fps: float):
    """Grabs webcam frames, JPEG-encodes, sends to WSL bridge."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # DSHOW = fast on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[video] ERROR: cannot open camera index {camera_index}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[video] Camera {camera_index} opened  {w}x{h}")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    interval = 1.0 / fps

    while True:
        print(f"[video] Connecting to {wsl_ip}:{VIDEO_PORT} ...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(5)
            sock.connect((wsl_ip, VIDEO_PORT))
            sock.settimeout(None)
            print("[video] Connected to WSL bridge.")

            while True:
                t0 = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    print("[video] Camera read failed")
                    break

                # Local preview window — press Q to quit
                cv2.imshow("Webcam (sending to WSL)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)

                ok, buf = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                if not send_msg(sock, buf.tobytes()):
                    print("[video] Send failed — reconnecting")
                    break

                elapsed = time.monotonic() - t0
                sleep = interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        except (ConnectionRefusedError, OSError) as e:
            print(f"[video] {e} — retrying in 2s")
        finally:
            try:
                sock.close()
            except Exception:
                pass

        time.sleep(2)


# ═══════════════════════════════════════════════════════════════
# Point cloud receiver thread
# ═══════════════════════════════════════════════════════════════

_cloud_queue: deque = deque(maxlen=2)


def cloud_receiver(wsl_ip: str):
    """Connects to WSL bridge and reads point cloud blobs."""
    while True:
        print(f"[cloud] Connecting to {wsl_ip}:{CLOUD_PORT} ...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((wsl_ip, CLOUD_PORT))
            sock.settimeout(None)
            print("[cloud] Connected to WSL bridge.")

            while True:
                data = recv_msg(sock)
                if data is None:
                    print("[cloud] Disconnected — reconnecting")
                    break
                xyz, rgb = unpack_cloud(data)
                if xyz is not None:
                    _cloud_queue.append((xyz, rgb))

        except (ConnectionRefusedError, OSError) as e:
            print(f"[cloud] {e} — retrying in 2s")
        finally:
            try:
                sock.close()
            except Exception:
                pass

        time.sleep(2)


# ═══════════════════════════════════════════════════════════════
# Open3D visualizer  (must run on main thread on Windows)
# ═══════════════════════════════════════════════════════════════

def run_open3d_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window("RTAB-Map Point Cloud  (live from WSL)", width=900, height=700)

    pcd = o3d.geometry.PointCloud()
    geometry_added = False

    print("[cloud] Open3D window open — waiting for point clouds from RTAB-Map ...")

    while True:
        # Grab latest cloud from queue
        latest = None
        while _cloud_queue:
            try:
                latest = _cloud_queue.popleft()
            except IndexError:
                break

        if latest is not None:
            xyz, rgb = latest
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

            if not geometry_added:
                vis.add_geometry(pcd)
                geometry_added = True
                ctl = vis.get_view_control()
                ctl.set_zoom(0.5)
            else:
                vis.update_geometry(pcd)

        if not vis.poll_events():   # returns False when window is closed
            break
        vis.update_renderer()
        time.sleep(0.033)           # ~30 fps UI refresh

    vis.destroy_window()


def run_fallback_visualizer():
    """Text stats fallback if Open3D is not installed."""
    print("[cloud] Fallback mode — printing cloud stats every 2s")
    while True:
        time.sleep(2)
        try:
            xyz, _ = _cloud_queue[-1]
            print(
                f"[cloud] {len(xyz):,} points  "
                f"x=[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]  "
                f"y=[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]  "
                f"z=[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]"
            )
        except (IndexError, TypeError):
            print("[cloud] No cloud received yet ...")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Windows webcam → WSL RTAB-Map bridge with live point-cloud viewer"
    )
    parser.add_argument(
        "--wsl-ip", default=None,
        help="WSL IP address (optional — auto-detected if not given)")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Windows camera index (default 0)")
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Send rate in Hz (default 10)")
    args = parser.parse_args()

    # Auto-detect WSL IP if not provided
    if args.wsl_ip is None:
        args.wsl_ip = get_wsl_ip()

    print("=" * 60)
    print("Windows → WSL RTAB-Map Bridge")
    print(f"  WSL IP   : {args.wsl_ip}")
    print(f"  Camera   : {args.camera}")
    print(f"  FPS      : {args.fps}")
    print(f"  Open3D   : {'yes' if OPEN3D else 'no  (pip install open3d)'}")
    print("=" * 60)
    print("Press Q in the webcam window to quit.")
    print()

    # Background threads
    threading.Thread(
        target=video_sender,
        args=(args.wsl_ip, args.camera, args.fps),
        daemon=True
    ).start()

    threading.Thread(
        target=cloud_receiver,
        args=(args.wsl_ip,),
        daemon=True
    ).start()

    # Visualizer must be on main thread (Windows requirement)
    time.sleep(1)
    if OPEN3D:
        run_open3d_visualizer()
    else:
        run_fallback_visualizer()


if __name__ == "__main__":
    main()