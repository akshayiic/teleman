"""Turn a portrait photo into a 1920x1080 AVTR-1 reference frame. Run inside the avtr-1
checkout, after `pixi run download`:

    pixi run -e renderer python /path/to/make_avtr1_portrait.py <photo> <out.png> --keep-background
    pixi run -e renderer python /path/to/make_avtr1_portrait.py <photo> <out.png> [--preview <preview.png>]

--keep-background: the whole photo scaled to 1080 high, with its left/right edge columns
repeated to fill 16:9 (seamless for plain or vertically graded backdrops). Saved without
alpha, so the renderer skips matting and only re-renders the face: the result looks
exactly like the photo.

Default: like the built-in avatars, an RGBA cut-out (renderer's own MODNet model) with the
head near the top and shoulders to the bottom edge, shown over AVTR1_BG. Works best with
a front-facing photo, neutral expression and closed mouth, on a plain background.
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

W, H = 1920, 1080
MODNET = os.environ.get("AVTR1_MODNET", "artifacts/main/build_artifacts/modnet.onnx")  # default storage layout
HEAD_TOP = 50  # built-in avatars: hair top at y ~40-160
HEAD_HEIGHT = 620  # hair top to chin in px, as in the built-in maria.png


def matte(rgb):
    """MODNet alpha in [0, 1], same preprocessing as avtr1_renderer.components.matting."""
    session = ort.InferenceSession(MODNET, providers=["CPUExecutionProvider"])
    x = cv2.resize(rgb, (512, 288), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    alpha = session.run(None, {"input": x})[0][0, 0]
    return cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("photo")
    p.add_argument("out")
    p.add_argument("--preview")
    p.add_argument("--keep-background", action="store_true")
    a = p.parse_args()

    photo = cv2.cvtColor(cv2.imread(a.photo, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ph, pw = photo.shape[:2]

    if a.keep_background:
        img = cv2.resize(photo, (round(pw * H / ph), H), interpolation=cv2.INTER_AREA)
        pad = max(0, W - img.shape[1])
        out = cv2.copyMakeBorder(img, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_REPLICATE)[:, :W]
        cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"photo {pw}x{ph} -> {W}x{H}, background kept")
        return

    # Person bounds in the photo (matte on a 16:9 letterbox so MODNet sees normal proportions).
    side = max(0, (ph * 16 // 9 - pw) // 2)
    box = cv2.copyMakeBorder(photo, 0, 0, side, side, cv2.BORDER_REPLICATE)
    a0 = matte(box)[:, side : side + pw]
    rows = np.where((a0 > 0.5).any(axis=1))[0]
    top = rows.min()
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face.detectMultiScale(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY), 1.1, 5, minSize=(pw // 8, pw // 8))
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    chin = fy + fh * 1.05
    cx = fx + fw / 2

    scale = HEAD_HEIGHT / (chin - top)
    scale = min(scale, H * 1.6 / ph)
    sw, sh = round(pw * scale), round(ph * scale)
    big = cv2.resize(photo, (sw, sh), interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)
    ox = round(W / 2 - cx * scale)
    oy = round(HEAD_TOP - top * scale)

    # Place on the canvas; fill outside the photo by edge replication so the matte stays clean.
    canvas = np.zeros((H, W, 3), np.uint8)
    inside = np.zeros((H, W), np.float32)
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + sw, W), min(oy + sh, H)
    canvas[y0:y1, x0:x1] = big[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    inside[y0:y1, x0:x1] = 1
    l, r, t, b = x0, W - x1, y0, H - y1
    canvas = cv2.copyMakeBorder(canvas[y0:y1, x0:x1], t, b, l, r, cv2.BORDER_REPLICATE)

    alpha = matte(canvas) * inside
    alpha = np.where(alpha < 0.04, 0, alpha)
    out = np.dstack([canvas, (alpha * 255).round().astype(np.uint8)])
    cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))
    ys, xs = np.where(alpha > 0.5)
    print(f"photo {pw}x{ph} scale {scale:.2f} -> person x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}")

    if a.preview:  # on a mid-grey backdrop, like the renderer's matting fixture
        prev = canvas * alpha[..., None] + 200 * (1 - alpha[..., None])
        cv2.imwrite(a.preview, cv2.cvtColor(prev.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
