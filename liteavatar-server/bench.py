"""Measures whether LiteAvatar renders faster than real time on this CPU.

Usage: vendor/venv/Scripts/python.exe bench.py <avatar_dir> <wav_16k_mono>
"""

import os
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "vendor", "engine")
sys.path.insert(0, ENGINE)
os.chdir(ENGINE)  # the engine loads its weights with relative paths

import torch  # noqa: E402

from lite_avatar import liteAvatar  # noqa: E402

avatar_dir, wav_path = sys.argv[1], sys.argv[2]
torch.set_num_threads(os.cpu_count() or 4)

t = time.time()
avatar = liteAvatar(data_dir=avatar_dir, num_threads=1, generate_offline=False)
avatar.load_dynamic_model(avatar_dir)
print(f"load: {time.time() - t:.1f}s")

with wave.open(wav_path, "rb") as w:
    assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16 kHz mono"
    pcm = w.readframes(w.getnframes())
seconds = len(pcm) / 32000

t = time.time()
params = avatar.audio2param(pcm)
t_params = time.time() - t
print(f"audio {seconds:.2f}s -> {len(params)} frames of mouth params in {t_params:.2f}s")

t = time.time()
n = min(len(params), 90)
for i in range(n):
    img = avatar.param2img(params[i], i % avatar.bg_video_frame_count)
    avatar.merge_mouth_to_bg(img, i % avatar.bg_video_frame_count)
t_frames = time.time() - t
fps = n / t_frames
print(f"rendered {n} frames in {t_frames:.2f}s = {fps:.1f} fps (real time needs 30)")
print(f"real-time factor (render+params / audio): {(t_params + len(params) / fps) / seconds:.2f} (<1 is faster than real time)")
