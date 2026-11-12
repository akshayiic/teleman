"""Local, free, unlimited talking-avatar server.

Text -> speech with Piper (offline TTS) -> mouth movements with LiteAvatar
(github.com/HumanAIGC/lite-avatar) -> rendered video frames, all on the CPU.

The browser POSTs {text, avatar, lang} to /speak and reads back a binary stream, one segment
per sentence, so playback can start before the whole text is rendered:

    each message = 1 byte type + 4 byte big-endian length + payload
    'S' segment header (JSON: fps, frames, startFrame); segments are a few seconds long
    'A' segment audio (WAV, Piper's native sample rate)
    'F' one video frame (JPEG), in order
    'E' end of stream (JSON: error message, if any)

Run with the project's bundled Python:
    liteavatar-server/vendor/venv/Scripts/python.exe liteavatar-server/server.py
"""

import io
import json
import os
import struct
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request
import uuid
from urllib.parse import parse_qs, quote, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
ENGINE = os.path.join(VENDOR, "engine")
# Faces from the LiteAvatarGallery (modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery),
# unzipped into vendor/avatars/<id>. Only one is held in memory at a time (~1 GB each);
# the browser picks one per request and switching reloads it (a few seconds).
AVATARS = {
    "priya": "P1i_ObCDaMOqyMuhLgKVy66Q",
    "kavya": "P1SVSj4XaExBEmdIUzCuDcPw",
}
# Faces from a single still photo, animated from a LivePortrait atlas (portrait_avatar.py).
PHOTOS = {
    "priti": os.path.join(HERE, "..", "public", "avatars", "priya.png"),
}


def env_setting(name, default=""):
    """From the environment, else from the app's .env.local (so `npm run liteavatar` picks it up)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        with open(os.path.join(HERE, "..", ".env.local"), encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep and key.strip() == name and value.strip():
                    return value.strip().strip("\"'")
    except OSError:
        pass
    return default


# Faces rendered by AVTR-1 on a GPU machine, through avtr1_bridge.py running there
# (reached at AVTR1_URL, e.g. a devtunnel). Speech is still made here with Piper.
AVTR1_URL = env_setting("AVTR1_URL").rstrip("/")
REMOTE = {
    "priti-hd": env_setting("AVTR1_AVATAR", "priya"),  # avatar id = PNG name in AVTR-1's reference_frames
    "aditi-hd": "aditi",
    "sahil-hd": "sahil",
}
DEFAULT_AVATAR = os.environ.get("LITEAVATAR_AVATAR", "priya")
# Piper voices (huggingface.co/rhasspy/piper-voices), by language.
VOICES = {
    "en": "en_US-lessac-medium",
    "hi": "hi_IN-priyamvada-medium",
}
# Men's faces speak with these instead.
MALE_FACES = {"sahil-hd"}
MALE_VOICES = {
    "en": "en_US-ryan-high",
    "hi": "hi_IN-pratham-medium",
}
PORT = int(os.environ.get("LITEAVATAR_PORT", "8765"))

FPS = 25  # output frame rate; the avatar's source video is 30 fps
WORKERS = 8  # parallel frame renderers (measured fastest on a 10-core i7)
OUTPUT_HEIGHT = 1280  # frames are scaled down from 1920 to keep the stream light
JPEG_QUALITY = 80

sys.path.insert(0, ENGINE)
os.chdir(ENGINE)  # the engine loads its weights with relative paths

import cv2  # noqa: E402
import torch  # noqa: E402
from piper import PiperVoice  # noqa: E402

from lite_avatar import liteAvatar  # noqa: E402

import portrait_avatar  # noqa: E402

torch.set_num_threads(1)  # parallelism comes from WORKERS instead


def disable_power_throttling():
    """Windows 11 runs background processes (no window) in "EcoQoS" mode on the
    efficiency cores at low clocks, which made rendering 4x slower after a
    short burst. Opt this process out (no admin rights needed)."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS
    state = PowerThrottlingState(1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)  # control speed, state off
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    process = kernel32.GetCurrentProcess()
    ok = kernel32.SetProcessInformation(process, ProcessPowerThrottling, ctypes.byref(state), ctypes.sizeof(state))
    # Also ask for above-normal priority so the renderer keeps the fast cores.
    kernel32.SetPriorityClass(process, 0x00008000)  # ABOVE_NORMAL_PRIORITY_CLASS
    print("power throttling disabled" if ok else f"could not disable power throttling ({ctypes.get_last_error()})")


disable_power_throttling()


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def avatar_dir(name):
    return os.path.join(VENDOR, "avatars", AVATARS[name])


log("loading LiteAvatar…")
avatar = liteAvatar(num_threads=1, fps=FPS, generate_offline=False)
loaded_avatar = None
log("loading Piper voices…")
voices = {lang: PiperVoice.load(os.path.join(VENDOR, "voices", f"{name}.onnx")) for lang, name in VOICES.items()}
male_voices = {lang: PiperVoice.load(os.path.join(VENDOR, "voices", f"{name}.onnx")) for lang, name in MALE_VOICES.items()}
render_pool = ThreadPoolExecutor(WORKERS)
# One speech at a time: rendering already uses every core.
speak_lock = threading.Lock()
# Held while a face is loaded or rendered, so switching faces never happens mid-speech.
model_lock = threading.Lock()


def load_photo(name):
    """Photo face, (re)building its LivePortrait atlas first if the photo is new or changed (~2 min)."""
    atlas = os.path.join(VENDOR, "avatars", f"atlas-{name}")
    if not portrait_avatar.atlas_is_current(PHOTOS[name], atlas):
        log(f"building the face atlas for {name} (one-off, ~2 min)…")
        portrait_avatar.build(PHOTOS[name], atlas, log=lambda msg: log(f"  {name}: {msg}"))
    return portrait_avatar.PortraitAvatar(PHOTOS[name], atlas, OUTPUT_HEIGHT)


photos = {name: load_photo(name) for name in PHOTOS}
FACES = sorted([*AVATARS, *PHOTOS, *REMOTE])


def bridge(path, data=None, timeout=15):
    """Request to the AVTR-1 bridge; raises with a readable message if it isn't reachable."""
    if not AVTR1_URL:
        raise RuntimeError("AVTR-1 isn't set up: put AVTR1_URL=<bridge or devtunnel URL> in .env.local and restart.")
    req = urllib.request.Request(AVTR1_URL + path, data, {"X-Tunnel-Skip-AntiPhishing-Page": "true"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            message = json.loads(e.read())["error"]
        except Exception:
            message = f"HTTP {e.code}"
        raise RuntimeError(f"AVTR-1 bridge: {message}") from e
    except OSError as e:
        raise RuntimeError(f"Can't reach the AVTR-1 bridge at {AVTR1_URL} ({e}).") from e


def read_messages(stream):
    """Messages (kind, payload) from a stream in our 1+4+payload framing."""
    while True:
        head = stream.read(5)
        if len(head) < 5:
            return
        payload = stream.read(struct.unpack(">I", head[1:])[0])
        yield head[:1].decode(), payload


remote_idle = {}


def use_avatar(name):
    """Make `name` the loaded face (call with model_lock held)."""
    global loaded_avatar
    if name in photos or name in REMOTE:
        return  # always ready; LiteAvatar's loaded face is kept
    if loaded_avatar != name:
        log(f"loading face {name} ({AVATARS[name]})…")
        avatar.load_dynamic_model(avatar_dir(name))
        loaded_avatar = name
        log(f"face {name} ready")


def preload(name):
    """Load a face in the background, e.g. as soon as the browser shows it."""

    def run():
        with model_lock:
            use_avatar(name)

    threading.Thread(target=run, daemon=True).start()


def make_photo_idle_clip(name):
    """Idle clip for a photo face: the still with its breathing motion, one loop
    of the same 150-frame timeline the speaking frames use. Rebuilt when the photo changes."""
    photo = photos[name]
    folder = os.path.join(VENDOR, "avatars", f"photo-{name}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "idle.webm")
    if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(os.path.join(folder, "..", f"atlas-{name}", "atlas.json")):
        first = photo.frame(portrait_avatar.REST_LEVEL, 0)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP80"), 30, first.shape[1::-1])
        for i in range(portrait_avatar.CLIP_FRAMES):
            out.write(photo.frame(portrait_avatar.REST_LEVEL, i))
        out.release()
    return path


def make_idle_clip(name):
    """Browser-playable (WebM/VP8) copy of the avatar's idle clip, sized like the
    speaking frames; the original is MPEG-4 Part 2, which browsers can't play."""
    if name in photos:
        return make_photo_idle_clip(name)
    path = os.path.join(avatar_dir(name), "idle.webm")
    if not os.path.exists(path):
        src = cv2.VideoCapture(os.path.join(avatar_dir(name), "bg_video.mp4"))
        out = None
        while True:
            ok, img = src.read()
            if not ok:
                break
            h, w = img.shape[:2]
            size = (round(w * OUTPUT_HEIGHT / h), OUTPUT_HEIGHT)
            out = out or cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP80"), 30, size)
            out.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
        if out:
            out.release()
    return path


IDLE_CLIPS = {name: make_idle_clip(name) for name in [*AVATARS, *PHOTOS]}
preload(DEFAULT_AVATAR)


def bg_frame(i, n):
    """Source background frame for output frame i (ping-pong through an n-frame 30 fps clip)."""
    i = int(i) % (2 * n)
    return i if i < n else 2 * n - 1 - i


def jpeg(img):
    h, w = img.shape[:2]
    if h != OUTPUT_HEIGHT:
        img = cv2.resize(img, (round(w * OUTPUT_HEIGHT / h), OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpg.tobytes()


def render_photo(photo, level, frame):
    return jpeg(photo.frame(level, frame))


def render(params, frame):
    with torch.inference_mode():
        mouth = avatar.param2img(params, frame)
        full, _ = avatar.merge_mouth_to_bg(mouth, frame)
    return jpeg(full)


def detect_lang(text):
    """Hindi if the text has any Devanagari, else English."""
    return "hi" if any("ऀ" <= c <= "ॿ" for c in text) else "en"


def speech_segments(text, voice, min_seconds=2.0, pad_seconds=1.2):
    """Piper audio per sentence, merging short sentences (LiteAvatar needs about
    a second of audio per call) and padding a short tail with silence."""
    pending, rate = [], 22050
    for chunk in voice.synthesize(text):
        rate = chunk.sample_rate
        pending.append(chunk.audio_int16_array)
        if sum(len(p) for p in pending) >= min_seconds * rate:
            yield np.concatenate(pending), rate
            pending = []
    if pending:
        pcm = np.concatenate(pending)
        short = int(pad_seconds * rate) - len(pcm)
        yield (np.pad(pcm, (0, short)) if short > 0 else pcm), rate


def wav_bytes(pcm16, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def resample_16k(pcm16, rate):
    """LiteAvatar's audio features expect 16 kHz mono PCM16."""
    if rate == 16000:
        return pcm16
    x = pcm16.astype(np.float32)
    n = int(len(x) * 16000 / rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console quiet
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
        name = parse_qs(url.query).get("avatar", [DEFAULT_AVATAR])[0]
        if name not in FACES:
            return self.send_json(404, {"error": f"Unknown avatar {name!r}; choose one of {FACES}"})
        if name in REMOTE and url.path in ("/health", "/idle.webm"):
            return self.remote_get(url.path, name)
        if url.path == "/":
            return self.send_json(
                200,
                {
                    "service": "LiteAvatar renderer for the teleman app",
                    "status": "running",
                    "open_the_app_at": "http://localhost:3000 (start it with: npm run dev)",
                },
            )
        if url.path == "/health":
            # The browser checks this when a face is shown: start loading it now.
            ready = name in photos or loaded_avatar == name
            if not ready:
                preload(name)
            return self.send_json(200, {"ready": ready, "fps": FPS, "sourceFps": 30, "languages": sorted(VOICES)})
        if url.path == "/idle.webm":
            # The avatar's own idle clip: natural motion with the mouth closed.
            data = open(IDLE_CLIPS[name], "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "video/webm")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_json(404, {"error": "not found"})

    def remote_get(self, path, name):
        """Health and idle clip of an AVTR-1 face, from the bridge on the GPU machine."""
        avatar_id = quote(REMOTE[name])
        try:
            if path == "/health":
                bridge(f"/health?avatar={avatar_id}").close()
                return self.send_json(200, {"ready": True, "fps": FPS, "sourceFps": 30, "languages": sorted(VOICES)})
            if name not in remote_idle:  # rendered from silence on the GPU; takes a few seconds the first time
                with bridge(f"/idle.webm?avatar={avatar_id}", timeout=180) as r:
                    remote_idle[name] = r.read()
        except RuntimeError as e:
            return self.send_json(503, {"error": str(e)})
        data = remote_idle[name]
        self.send_response(200)
        self.send_header("Content-Type", "video/webm")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")  # the face on the GPU machine can change
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/speak":
            return self.send_json(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = str(body.get("text", "")).strip()[:2000]
        start_frame = float(body.get("startFrame", 0))  # in source (30 fps) frames, to continue the idle clip
        name = str(body.get("avatar", DEFAULT_AVATAR))
        lang = str(body.get("lang") or detect_lang(text))
        if not text:
            return self.send_json(400, {"error": "Type something to say."})
        if name not in FACES:
            return self.send_json(400, {"error": f"Unknown avatar {name!r}; choose one of {FACES}"})
        if lang not in voices:
            return self.send_json(400, {"error": f"Unknown language {lang!r}; choose one of {sorted(voices)}"})
        if not speak_lock.acquire(blocking=False):
            return self.send_json(409, {"error": "Already speaking; stop first."})

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(kind, payload):
            msg = kind.encode() + struct.pack(">I", len(payload)) + payload
            self.wfile.write(f"{len(msg):x}\r\n".encode() + msg + b"\r\n")
            self.wfile.flush()

        error = None
        futures = []
        model_lock.acquire()
        try:
            use_avatar(name)
            photo = photos.get(name)
            step = 30 / FPS
            session = uuid.uuid4().hex  # AVTR-1 motion continues across this speech's segments
            voice = (male_voices if name in MALE_FACES else voices)[lang]
            for pcm, rate in speech_segments(text, voice):
                t = time.time()
                if name in REMOTE:
                    pcm16 = resample_16k(pcm, rate).tobytes()
                    frames = -(-len(pcm16) // 6400) * 5  # the bridge renders 5 frames per 0.2 s of audio
                    # Connect before announcing the segment, so an unreachable bridge fails cleanly.
                    with bridge(f"/render?avatar={quote(REMOTE[name])}&session={session}", pcm16, timeout=120) as r:
                        send("S", json.dumps({"fps": FPS, "frames": frames, "startFrame": start_frame}).encode())
                        send("A", wav_bytes(pcm, rate))
                        for kind, payload in read_messages(r):
                            if kind == "F":
                                send("F", payload)
                            elif kind == "E" and json.loads(payload)["error"]:
                                raise RuntimeError("AVTR-1: " + json.loads(payload)["error"])
                    start_frame += frames * step
                    log(f"segment {len(pcm) / rate:.1f}s audio, {frames} AVTR-1 frames in {time.time() - t:.1f}s")
                    continue
                if photo:
                    n = portrait_avatar.CLIP_FRAMES
                    jobs = [(render_photo, photo, lvl) for lvl in portrait_avatar.mouth_levels(resample_16k(pcm, rate), FPS)]
                else:
                    n = avatar.bg_video_frame_count
                    params = avatar.audio2param(resample_16k(pcm, rate).tobytes(), is_complete=True)
                    jobs = [(render, p) for p in params]
                send("S", json.dumps({"fps": FPS, "frames": len(jobs), "startFrame": start_frame}).encode())
                send("A", wav_bytes(pcm, rate))
                futures = [
                    render_pool.submit(*job, bg_frame(start_frame + i * step, n)) for i, job in enumerate(jobs)
                ]
                for f in futures:  # stream in order as frames finish
                    send("F", f.result())
                start_frame += len(jobs) * step
                log(f"segment {len(pcm) / rate:.1f}s audio, {len(jobs)} frames in {time.time() - t:.1f}s")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            log("client stopped")
            for f in futures:
                f.cancel()
            return
        except Exception as e:  # report to the browser instead of hanging up
            error = str(e)
            log("error:", error)
        finally:
            model_lock.release()
            speak_lock.release()
        try:
            send("E", json.dumps({"error": error}).encode())
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass


class Server(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second copy bind the same port and silently
    # share requests with a stale one; fail with "address in use" instead.
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        # A browser that stops listening mid-stream is normal, not an error.
        if not isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            super().handle_error(request, client_address)


if __name__ == "__main__":
    server = Server(("127.0.0.1", PORT), Handler)
    log(f"ready on http://127.0.0.1:{PORT}")
    server.serve_forever()
