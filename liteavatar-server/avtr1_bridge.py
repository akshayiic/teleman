"""Bridge between the teleman app and an AVTR-1 renderer, run on the GPU machine.

AVTR-1 (github.com/avaturn-live/avtr-1) renders 5 frames per HTTP call and needs the
previous call's state blob, so a sentence is dozens of sequential calls. Doing those
over a tunnel from the laptop would be far too slow; this bridge makes them locally
and streams finished JPEG frames back, one request per sentence.

Run it inside the avtr-1 checkout, next to its renderer:

    pixi run python -m avtr1_renderer.api.app                 # renderer on :8000
    pixi run python /path/to/avtr1_bridge.py                  # bridge on :8770
    devtunnel host -p 8770 --allow-anonymous                  # or any other tunnel

Only needs numpy and OpenCV (both in the renderer's pixi environment).

    GET  /health?avatar=<id>        {"ready": true, "avatars": [...]} or 503 {"error": ...}
    GET  /idle.webm?avatar=<id>     ~6 s of the avatar idling (rendered from silence, cached)
    POST /render?avatar=<id>&session=<id>[&width=<px>][&patch=1[&key=0]]   body: int16 16 kHz mono PCM
         -> stream of messages, 1 byte type + 4 byte big-endian length + payload:
            'F' one frame (JPEG), 'E' end (JSON: error message, if any)
         Calls with the same session continue the motion of the previous call.
         patch=1 (for slow links): only the face moves, so the first frame comes as 'K'
         (full JPEG) and the rest as 'P' (2+2 byte big-endian x, y, then a JPEG of the
         face region to draw over 'K' at x, y): ~5x less data at full resolution. With key=0
         every frame is a 'P', for a client that keeps one 'K' as its fixed background.

Environment: AVTR1_RENDERER (default http://127.0.0.1:8000), AVTR1_BG (background id,
default: the renderer's first), BRIDGE_PORT (default 8770).
"""

import json
import os
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

RENDERER = os.environ.get("AVTR1_RENDERER", "http://127.0.0.1:8000").rstrip("/")
PORT = int(os.environ.get("BRIDGE_PORT", "8770"))
FPS = 25
CURRENT = 3200  # samples per call: 5 frames x 640 (renderer's chunk_size x frame_len)
FUTURE = 3280  # look-ahead samples per call: 5 x 640 + 80 audio shift
IDLE_SECONDS = 6
JPEG_QUALITY = 75  # ~115 KB per 720p frame; live lip sync streams 25 a second

sessions = {}  # session id -> (state blob, last used); motion continues across sentences
sessions_lock = threading.Lock()
render_lock = threading.Lock()  # one GPU, one stream at a time
idle_clips = {}
face_boxes = {}  # avatar -> (x, y, w, h) of the region AVTR-1 changes, for patch=1
PATCH_QUALITY = 90
KEY_QUALITY = 92  # the background a client may keep for a whole call


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def renderer_get(path):
    with urllib.request.urlopen(RENDERER + path, timeout=10) as r:
        return json.loads(r.read())


def background():
    bg = os.environ.get("AVTR1_BG")
    if bg:
        return bg
    found = renderer_get("/avatars")["backgrounds"]
    return found[0] if found else "default"


def multipart(fields):
    boundary = uuid.uuid4().hex
    body = b""
    for name, data in fields.items():
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.raw"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + data + b"\r\n"
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def render_chunk(avatar, bg, cur, fut, state):
    """One renderer call: 5 frames of BGR images plus the next state blob."""
    silence_cur, silence_fut = bytes(len(cur)), bytes(len(fut))
    fields = {
        "current_chunk": cur,
        "future_chunk": fut,
        "current_chunk_listen": silence_cur,
        "future_chunk_listen": silence_fut,
    }
    if state:
        fields["state"] = state
    body, ctype = multipart(fields)
    query = f"avatar_id={urllib.request.quote(avatar)}&bg_id={urllib.request.quote(bg)}&pixel_format=yuv_i420"
    req = urllib.request.Request(f"{RENDERER}/process-audio-v3?{query}", body, {"Content-Type": ctype})
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"AVTR-1 renderer error {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    with r:
        state_len = int(r.headers["X-State-Length-Bytes"])
        h, w = int(r.headers["X-Frame-Height"]), int(r.headers["X-Frame-Width"])
        data = r.read()
    size = h * w * 3 // 2
    frames = []
    for i in range((len(data) - state_len) // size):
        yuv = np.frombuffer(data, np.uint8, size, state_len + i * size).reshape(h * 3 // 2, w)
        frames.append(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420))
    return frames, data[:state_len]


def render(avatar, pcm, state=None):
    """Yields BGR frames for int16 PCM bytes; returns the final state blob via StopIteration."""
    bg = background()
    pcm = pcm + bytes((-len(pcm)) % (CURRENT * 2)) + bytes(FUTURE * 2)  # whole calls + look-ahead
    for start in range(0, len(pcm) - FUTURE * 2, CURRENT * 2):
        cur = pcm[start : start + CURRENT * 2]
        fut = pcm[start + CURRENT * 2 : start + (CURRENT + FUTURE) * 2]
        frames, state = render_chunk(avatar, bg, cur, fut, state)
        yield from frames
    return state


def idle_clip(avatar):
    if avatar not in idle_clips:
        path = os.path.join(tempfile.gettempdir(), f"avtr1-idle-{avatar}.webm")
        out = None
        with render_lock:
            for img in render(avatar, bytes(IDLE_SECONDS * 16000 * 2)):
                out = out or cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP80"), FPS, img.shape[1::-1])
                out.write(img)
        out.release()
        idle_clips[avatar] = open(path, "rb").read()
    return idle_clips[avatar]


def face_box(avatar):
    """The part of the frame that moves (face and head), measured once per avatar from a
    few seconds of speech-like audio, with a margin for bigger head movements."""
    if avatar not in face_boxes:
        rng = np.random.default_rng(0)
        t = np.arange(16000 * 3) / 16000
        pcm = (np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 9000 + rng.normal(0, 800, t.size))
        with render_lock:
            stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in render(avatar, pcm.astype(np.int16).tobytes())])
        moving = (cv2.GaussianBlur(stack.max(0) - stack.min(0), (0, 0), 3) > 18).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(moving)
        x, y, w, h = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), :4]
        H, W = stack.shape[1:]
        cx, cy, w, h = x + w / 2, y + h / 2, w * 2 + 48, h * 2 + 48
        x0, y0 = max(0, int(cx - w / 2)) // 8 * 8, max(0, int(cy - h / 2)) // 8 * 8
        x1, y1 = min(W, int(cx + w / 2 + 7) // 8 * 8), min(H, int(cy + h / 2 + 7) // 8 * 8)
        face_boxes[avatar] = (x0, y0, x1 - x0, y1 - y0)
        log(f"{avatar}: face region {face_boxes[avatar]} of {W}x{H}")
    return face_boxes[avatar]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        avatar = parse_qs(url.query).get("avatar", [""])[0]
        if url.path == "/health":
            try:
                renderer_get("/health")
                avatars = renderer_get("/avatars")["avatars"]
            except Exception as e:
                return self.send_json(503, {"error": f"AVTR-1 renderer at {RENDERER} isn't ready: {e}"})
            if avatar and avatar not in avatars:
                return self.send_json(
                    404, {"error": f"AVTR-1 has no avatar {avatar!r}; available: {avatars}", "avatars": avatars}
                )
            return self.send_json(200, {"ready": True, "avatars": avatars})
        if url.path == "/idle.webm":
            try:
                data = idle_clip(avatar)
            except Exception as e:
                return self.send_json(502, {"error": str(e)})
            self.send_response(200)
            self.send_header("Content-Type", "video/webm")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/render":
            return self.send_json(404, {"error": "not found"})
        q = parse_qs(url.query)
        avatar, session = q.get("avatar", [""])[0], q.get("session", [""])[0]
        width = int(q.get("width", ["0"])[0] or 0)  # optional smaller frames, for slow links
        try:
            box = face_box(avatar) if q.get("patch", [""])[0] == "1" else None
            want_key = q.get("key", ["1"])[0] != "0"
        except Exception as e:
            return self.send_json(502, {"error": f"AVTR-1: {e}"})
        pcm = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        pcm = pcm[: len(pcm) // 2 * 2]

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(kind, payload):
            msg = kind.encode() + struct.pack(">I", len(payload)) + payload
            self.wfile.write(f"{len(msg):x}\r\n".encode() + msg + b"\r\n")
            self.wfile.flush()

        error = None
        t = time.time()
        n = 0
        with render_lock:
            # Read the motion state only once it's our turn: the client may send the next piece
            # of the same session while this one is still rendering.
            with sessions_lock:
                now = time.time()
                for k in [k for k, (_, used) in sessions.items() if now - used > 600]:
                    del sessions[k]
                state = sessions.get(session, (None, 0))[0]
            try:
                frames = render(avatar, pcm, state)
                while True:
                    try:
                        img = next(frames)
                    except StopIteration as done:
                        if session:
                            with sessions_lock:
                                sessions[session] = (done.value, time.time())
                        break
                    if box and (n or not want_key):
                        x, y, w, h = box
                        jpeg = cv2.imencode(".jpg", img[y : y + h, x : x + w], [cv2.IMWRITE_JPEG_QUALITY, PATCH_QUALITY])[1]
                        send("P", struct.pack(">HH", x, y) + jpeg.tobytes())
                    elif box:
                        send("K", cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, KEY_QUALITY])[1].tobytes())
                    else:
                        if 0 < width < img.shape[1]:
                            img = cv2.resize(img, (width, img.shape[0] * width // img.shape[1]), interpolation=cv2.INTER_AREA)
                        send("F", cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes())
                    n += 1
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                log("client stopped")
                return
            except Exception as e:
                error = str(e)
                log("error:", error)
        log(f"{avatar}: {len(pcm) / 32000:.1f}s audio, {n} frames in {time.time() - t:.1f}s")
        try:
            send("E", json.dumps({"error": error}).encode())
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log(f"AVTR-1 bridge on http://0.0.0.0:{PORT}, renderer {RENDERER}")
    server.serve_forever()
