"""Talking presenter from a single still photo, for faces LiteAvatar has no trained model for.

LivePortrait (github.com/KwaiVGI/LivePortrait, MIT; ONNX export from
huggingface.co/warmshao/FasterLivePortrait) can re-render a portrait with the lips
opened to a given ratio or the eyes closed. At ~13 s per frame on this CPU it is far
too slow to run live, so `build` renders a handful of mouth states and a blink once
per photo (~2 min, cached), and `PortraitAvatar` composes frames at speaking time by
cross-fading the mouth state that matches the loudness of the audio. Only the mouth
and eye areas come from the renders; the rest stays the untouched photo.

A slow "breathing" zoom and blinks are keyed to the same 150-frame ping-pong timeline
as the LiteAvatar idle clips (source frames at 30 fps), so idle and speech line up.

    python portrait_avatar.py <photo.png> <atlas_dir>   # (re)build an atlas by hand
"""

import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "vendor", "liveportrait")

# Lip-open ratios (LivePortrait's lip_close_ratio target), closed -> wide open.
LIP_LEVELS = [0.0, 0.08, 0.16, 0.24, 0.32, 0.42, 0.52]
SOURCE_LIP = 0.25  # approx. lip ratio of a smiling source photo
SOURCE_EYE = 0.38  # approx. eye-open ratio of the source photo
CROP_SCALE, CROP_VY = 2.3, -0.125  # LivePortrait's crop config
RENDER = 512  # LivePortrait output size
CLIP_FRAMES = 150  # idle timeline length (5 s at 30 fps)
REST_LEVEL = 1.0  # mouth state while idle or silent: lips together, slight smile
BLINKS = (40, 115)  # timeline frames where a blink starts
BLINK_FRAMES = 7


# ---------------------------------------------------------------- atlas build


def face_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))
    if len(faces) == 0:
        raise ValueError("No face found in the photo; use a clear, front-facing portrait.")
    return max(faces, key=lambda f: f[2] * f[3])


def crop_square(img):
    """LivePortrait-style square crop around the face; returns (crop, x0, y0, side)."""
    x, y, w, h = face_box(img)
    size = (w + h) / 2
    side = int(size * CROP_SCALE)
    cx, cy = x + w / 2, y + h / 2 + CROP_VY * size
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    padded = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_REPLICATE)
    return padded[y0 + side : y0 + 2 * side, x0 + side : x0 + 2 * side], x0, y0, side


def _headpose_deg(bins):
    p = np.exp(bins - bins.max())
    p /= p.sum()
    return float((p * np.arange(66)).sum() * 3 - 97.5)


def _rotation(pitch, yaw, roll):
    x, y, z = np.radians([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return (rz @ ry @ rx).T


def _cpu_warping_model():
    """The ONNX export uses a TensorRT plugin op (GridSample3D); rewrite it as the
    standard opset-20 GridSample, which onnxruntime runs on the CPU."""
    out = os.path.join(MODELS, "warping_spade_cpu.onnx")
    if not os.path.exists(out):
        import onnx
        from onnx import version_converter

        m = onnx.load(os.path.join(MODELS, "warping_spade-fix.onnx"))
        for n in m.graph.node:
            if n.op_type == "GridSample3D":
                n.op_type = "GridSample"
        m = version_converter.convert_version(m, 20)
        for n in m.graph.node:
            if n.op_type == "GridSample":
                for a in n.attribute:
                    if a.name == "mode":
                        a.s = b"linear"
        onnx.save(m, out)
    return out


def build(photo_path, out_dir, log=print):
    import onnxruntime as ort

    def session(path):
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    os.makedirs(out_dir, exist_ok=True)
    crop, x0, y0, side = crop_square(cv2.imread(photo_path, cv2.IMREAD_COLOR))
    inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_AREA)
    inp = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    f_s = session(os.path.join(MODELS, "appearance_feature_extractor.onnx")).run(None, {"img": inp})[0]
    pitch, yaw, roll, t, exp, scale, kp = session(os.path.join(MODELS, "motion_extractor.onnx")).run(None, {"img": inp})
    R = _rotation(_headpose_deg(pitch[0]), _headpose_deg(yaw[0]), _headpose_deg(roll[0]))
    x_s = (kp.reshape(1, 21, 3) @ R + exp.reshape(1, 21, 3)) * scale[0, 0]
    x_s[..., :2] += t[:, None, :2]
    x_s = x_s.astype(np.float32)
    flat = x_s.reshape(1, 63)

    lip = session(os.path.join(MODELS, "stitching_lip.onnx"))
    eye = session(os.path.join(MODELS, "stitching_eye.onnx"))
    warp = session(_cpu_warping_model())

    def render(x_d, name):
        out = warp.run(None, {"feature_3d": f_s, "kp_driving": x_d, "kp_source": x_s})[0][0]
        rgb = (out.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        log(f"rendered {name}")

    for i, level in enumerate(LIP_LEVELS):
        d = lip.run(None, {"input": np.concatenate([flat, [[SOURCE_LIP, level]]], 1).astype(np.float32)})[0]
        render(x_s + d.reshape(1, 21, 3), f"lip_{i}.png")
    d = eye.run(None, {"input": np.concatenate([flat, [[SOURCE_EYE, SOURCE_EYE, 0.0]]], 1).astype(np.float32)})[0]
    render(x_s + d.reshape(1, 21, 3), "eyes_closed.png")

    meta = {"x0": x0, "y0": y0, "side": side, "levels": LIP_LEVELS, "photo_mtime": os.path.getmtime(photo_path)}
    with open(os.path.join(out_dir, "atlas.json"), "w") as f:
        json.dump(meta, f)


def atlas_is_current(photo_path, atlas_dir):
    try:
        with open(os.path.join(atlas_dir, "atlas.json")) as f:
            meta = json.load(f)
    except OSError:
        return False
    return meta["photo_mtime"] == os.path.getmtime(photo_path) and meta["levels"] == LIP_LEVELS


# ---------------------------------------------------------------- runtime


def _ellipse_mask(side, center, axes):
    mask = np.zeros((side, side), np.float32)
    cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), 0, 0, 360, 1, -1)
    k = max(3, (int(min(axes)) // 2) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)[..., None]


class PortraitAvatar:
    def __init__(self, photo_path, atlas_dir, height):
        meta = json.load(open(os.path.join(atlas_dir, "atlas.json")))
        img = cv2.imread(photo_path, cv2.IMREAD_COLOR)
        x0, y0, side = meta["x0"], meta["y0"], meta["side"]
        # The crop may stick out of the photo; only its part inside the photo is used.
        H, W = img.shape[:2]
        self.region = (max(y0, 0), min(y0 + side, H), max(x0, 0), min(x0 + side, W))
        ry1, ry2, rx1, rx2 = self.region
        cy1, cx1 = ry1 - y0, rx1 - x0  # where that part starts inside the crop
        inside = (slice(cy1, cy1 + ry2 - ry1), slice(cx1, cx1 + rx2 - rx1))
        base = img[ry1:ry2, rx1:rx2].astype(np.float32)

        # Face geometry inside the crop (see crop_square): face size and the box's top edge.
        s = side / CROP_SCALE
        top = side / 2 - CROP_VY * s - s / 2
        mouth = _ellipse_mask(side, (side / 2, top + 0.80 * s), (0.40 * s, 0.30 * s))[inside]
        eyes = _ellipse_mask(side, (side / 2, top + 0.40 * s), (0.44 * s, 0.13 * s))[inside]

        def load(name):
            full = cv2.resize(cv2.imread(os.path.join(atlas_dir, name)), (side, side), interpolation=cv2.INTER_CUBIC)
            return full[inside].astype(np.float32)

        # Pre-blended crops: the photo with each mouth state (and the closed eyes) pasted in.
        self.mouths = [base * (1 - mouth) + load(f"lip_{i}.png") * mouth for i in range(len(LIP_LEVELS))]
        self.eyes_closed = load("eyes_closed.png")
        self.eye_mask = eyes
        self.image = img
        self.height = height

    def frame(self, level, timeline):
        """Full BGR frame with the mouth at `level` (0..len(LIP_LEVELS)-1, fractional cross-fades)
        at source frame `timeline` of the idle loop (blinks + breathing)."""
        lo = int(min(max(level, 0), len(self.mouths) - 1))
        hi = min(lo + 1, len(self.mouths) - 1)
        a = min(max(level - lo, 0), 1)
        crop = self.mouths[lo] * (1 - a) + self.mouths[hi] * a if a else self.mouths[lo]
        b = blink_amount(timeline)
        if b:
            crop = crop * (1 - self.eye_mask * b) + self.eyes_closed * (self.eye_mask * b)
        img = self.image.copy()
        ry1, ry2, rx1, rx2 = self.region
        img[ry1:ry2, rx1:rx2] = crop.clip(0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        if h != self.height:
            img = cv2.resize(img, (round(w * self.height / h), self.height), interpolation=cv2.INTER_AREA)
        return cv2.warpAffine(img, motion(timeline, img.shape), img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def blink_amount(timeline):
    """0 = eyes open, 1 = closed; a quick close and slightly slower open."""
    f = timeline % CLIP_FRAMES
    for start in BLINKS:
        d = f - start
        if 0 <= d < BLINK_FRAMES:
            return math.sin(math.pi * (d / BLINK_FRAMES) ** 0.7)
    return 0.0


def motion(timeline, shape):
    """Slow breathing zoom and sway for source frame `timeline`, as a 2x3 affine matrix."""
    h, w = shape[:2]
    t = (timeline % CLIP_FRAMES) / CLIP_FRAMES
    m = cv2.getRotationMatrix2D((w / 2, h * 0.35), 0.25 * math.sin(2 * math.pi * t + 1.3), 1.012 + 0.008 * math.sin(2 * math.pi * t))
    m[1, 2] += 2.0 * math.sin(2 * math.pi * t + 0.6)
    return m


def mouth_levels(pcm16_16k, fps):
    """Mouth state per video frame from the speech audio: louder = more open, hissy
    sounds (s, sh, f) less open, silence back to rest; mouth leads the sound slightly."""
    x = pcm16_16k.astype(np.float32) / 32768.0
    hop = 16000 // fps
    n = max(1, len(x) // hop)
    win = np.hanning(2 * hop).astype(np.float32)
    padded = np.pad(x, (hop, 2 * hop))
    freqs = np.fft.rfftfreq(2 * hop, 1 / 16000)
    db, hiss = np.zeros(n), np.zeros(n)
    for i in range(n):
        spec = np.abs(np.fft.rfft(padded[i * hop : i * hop + 2 * hop] * win)) ** 2
        total = spec.sum() + 1e-9
        db[i] = 10 * np.log10(total / len(win) + 1e-10)
        hiss[i] = spec[freqs > 3500].sum() / total
    # TTS speech is evenly loud, so scale against this clip's own range and add the
    # syllable rhythm (loudness above its local average) to open/close per syllable.
    lo, hi = np.percentile(db, 30), np.percentile(db, 97)
    voiced = db > lo - 6
    loud = np.clip((db - lo) / max(hi - lo, 1e-3), 0, 1)
    local = np.convolve(np.pad(db, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    syllable = np.clip((db - local) / 8 + 0.5, 0, 1)
    open_ = (0.6 + 2.4 * loud + 2.0 * syllable * loud) * (1 - 0.6 * hiss)
    target = np.where(voiced, np.clip(open_, 0, len(LIP_LEVELS) - 1.5), REST_LEVEL)
    # Fast attack and release, then lead the audio by one frame.
    out, cur = np.zeros(n), REST_LEVEL
    for i, v in enumerate(target):
        cur += (v - cur) * (0.7 if v > cur else 0.55)
        out[i] = cur
    return np.concatenate([out[1:], out[-1:]])


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
