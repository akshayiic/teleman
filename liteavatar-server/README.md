# LiteAvatar server ("Priya")

Free, unlimited, photoreal talking presenter rendered on this PC's CPU:

- **Voice:** [Piper](https://github.com/OHF-Voice/piper1-gpl), offline: English `en_US-lessac-medium`,
  Hindi `hi_IN-priyamvada-medium`. Without an explicit language, text with Devanagari is read in Hindi.
- **Photo faces ("priti"):** any still portrait listed in `PHOTOS` in `server.py`
  (`public/avatars/priya.png`). [LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT) is
  far too slow on this CPU to run live (~13 s/frame), so `portrait_avatar.py` uses it once per photo
  (~2 min at server start, cached in `vendor/avatars/atlas-<name>/`) to render 7 mouth-open levels
  and a blink. While speaking, the mouth level follows the audio's loudness/syllables and the
  renders are cross-faded into the untouched photo, plus blinks and a slight breathing motion.
  Replace the PNG and restart the server to change the face (front-facing, mouth visible).
- **Gallery faces ("priya" default, "kavya"):** [LiteAvatar](https://github.com/HumanAIGC/lite-avatar)
  (MIT) models listed in `AVATARS` in `server.py`. These are trained from video, so they move more
  naturally and shape the mouth per sound, but only gallery faces are available. One is held in
  memory at a time; switching reloads it (~4 s). To add one, unzip its gallery zip into `vendor/avatars/`.

Every face also needs an entry in `LOCAL` in `src/components/Stage.tsx`.

The Next.js app reaches it through the `/liteavatar/*` rewrite in `next.config.ts`.

## Run

```sh
npm run liteavatar   # in a second terminal, next to `npm run dev`
```

Then choose **Priya (real, free, local)** in the app.

## Performance

Rendering is CPU-bound (~40–70 ms per frame, 25 fps output). On the i7-13620H
laptop it was measured at ~25–30 fps in short bursts and ~12–14 fps sustained,
so long texts take up to ~2× their spoken length to render. The player buffers
before each segment and pauses the audio if a frame is late, so lips and voice
stay in sync; it just starts later. Plugging in and choosing Windows
"Best performance" power mode helps.

## Setup (`vendor/` is git-ignored, ~3 GB)

Done once from the project root with Git Bash. GitHub downloads are very slow
on the office network, so Python comes from NuGet and code via the GitHub API.

```sh
cd liteavatar-server && mkdir -p vendor && cd vendor

# Engine code
curl -L -o lite-avatar.zip https://api.github.com/repos/HumanAIGC/lite-avatar/zipball/main
unzip -q lite-avatar.zip && mv HumanAIGC-lite-avatar-* engine && rm lite-avatar.zip
mkdir -p avatars && unzip -q engine/data/sample_data.zip -d avatars/   # -> avatars/preload (sample)
# Gallery faces (each zip contains <id>/...):
for id in P1i_ObCDaMOqyMuhLgKVy66Q P1SVSj4XaExBEmdIUzCuDcPw; do
  curl -L -o a.zip https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/resolve/master/20250408/$id.zip
  unzip -q a.zip -d avatars/ && rm a.zip
done

# Weights (ModelScope; verify SHA-256 against the repo listing)
B=https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/resolve/master/lite_avatar_weights
W=engine/weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
curl -L -C - -o engine/weights/model_1.onnx $B/model_1.onnx
curl -L -C - -o $W/lm/lm.pb $B/lm.pb
curl -L -C - -o $W/model.pb $B/model.pb

# Portable Python 3.10 + virtualenv
curl -L -o python.nupkg https://www.nuget.org/api/v2/package/python/3.10.11
mkdir -p py && unzip -q python.nupkg -d py && rm python.nupkg
./py/tools/python.exe -m venv venv
P=./venv/Scripts/python.exe

# Dependencies (CPU PyTorch; numpy < 2 for torch 2.1)
$P -m pip install torch==2.1.1+cpu torchvision==0.16.1+cpu torchaudio==2.1.1+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
$P -m pip install "numpy<2" scipy==1.11.4 librosa==0.9.2 resampy==0.4.3 soundfile pydub \
  opencv-python==4.11.0.86 onnxruntime==1.20.1 loguru tqdm typeguard==2.13.3 PyYAML \
  humanfriendly torch-complex==0.4.4 editdistance sentencepiece jieba jaconv omegaconf \
  hydra-core einops numba==0.60.0 llvmlite==0.43.0 six jamo h5py oss2 modelscope piper-tts

# Piper voice
mkdir -p voices && V=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
curl -L -o voices/en_US-lessac-medium.onnx $V/en_US-lessac-medium.onnx
curl -L -o voices/en_US-lessac-medium.onnx.json $V/en_US-lessac-medium.onnx.json
V=https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium
curl -L -o voices/hi_IN-priyamvada-medium.onnx $V/hi_IN-priyamvada-medium.onnx
curl -L -o voices/hi_IN-priyamvada-medium.onnx.json $V/hi_IN-priyamvada-medium.onnx.json

# LivePortrait ONNX models, for photo faces (the server converts the warping model to a
# CPU-compatible copy, warping_spade_cpu.onnx, on first use; needs the onnx package)
$P -m pip install onnx==1.16.2
mkdir -p liveportrait && B=https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx
for f in appearance_feature_extractor motion_extractor stitching_lip stitching_eye warping_spade-fix; do
  curl -L -o liveportrait/$f.onnx $B/$f.onnx
done
```

`bench.py` measures render speed: `vendor/venv/Scripts/python.exe bench.py <avatar_dir> <16k_mono.wav>`.

## AVTR-1 on a GPU machine ("Priti HD")

[AVTR-1](https://github.com/avaturn-live/avtr-1) animates a photo with head motion and
expressions, close to Tavus, but needs Linux and an NVIDIA RTX 20-series or newer GPU
(real time from about an RTX 3060 Ti; a GTX 10-series card is not supported by its
CUDA 12.8 / TensorRT 10 stack). Its renderer and streamer code are non-commercial only.

The app keeps making the speech (Piper, English/Hindi) on this PC and sends the audio
to `avtr1_bridge.py` on the GPU machine, which drives the AVTR-1 renderer there and
streams the frames back (one request per sentence, so a tunnel's latency doesn't matter).

On the GPU machine (Linux):

```sh
git clone https://github.com/avaturn-live/avtr-1 && cd avtr-1
pixi install
pixi run download            # weights; gated: request access on huggingface.co/avaturn-live/avtr-1 first
pixi run build-trt-engines   # once per machine (~10 min on an RTX 5070)
cp /path/to/teleman/liteavatar-server/avtr1_bridge.py .
# Priti's photo as an AVTR-1 avatar (id = file name), 16:9 with its own background:
pixi run -e renderer python /path/to/teleman/liteavatar-server/make_avtr1_portrait.py --keep-background \
  /path/to/teleman/public/avatars/priya.png artifacts/main/avatars_artifacts/reference_frames/priya.png

bash /path/to/teleman/liteavatar-server/start-avtr1.sh   # renderer on :8000 + bridge on :8770
devtunnel host -p 8770 --allow-anonymous                 # if this PC can't reach the GPU machine directly
```

AVTR-1 renders 16:9 (1280x720), so a portrait photo copied in as-is gets squeezed.
`make_avtr1_portrait.py --keep-background` fits the whole photo to 1080 high and extends its
background sideways; as it has no alpha channel, the renderer leaves the photo untouched
and only animates the face, so Priti looks exactly like `priya.png`. Without the flag the
script makes a transparent cut-out framed like the built-in avatars (e.g. `maria.png`),
shown over the `AVTR1_BG` backdrop from `start-avtr1.sh` (default `minimal_office`; the
renderer's `/avatars` lists the others). Either way, lip sync is best from a front-facing
photo with a neutral expression and closed mouth.

The renderer logs a "Keep alive loop iteration failed" traceback every ~30 s because it
looks for AVTR-1's cloud load balancer; it doesn't affect rendering.

### Windows GPU machine (WSL)

On Windows, AVTR-1 runs in WSL 2 (tested: Ubuntu 24.04, RTX 5070, WSL 3.0.3):

- `wsl --install -d Ubuntu-24.04`, restart, then follow the Linux steps above inside Ubuntu.
  WSL distributions are per Windows user: install and run it under the same account.
- To reach the bridge from the LAN without a tunnel, set `networkingMode=mirrored` under
  `[wsl2]` in `%USERPROFILE%\.wslconfig` (then `wsl --shutdown`) and allow TCP 8770 inbound
  in the Windows firewall and the Hyper-V firewall (`New-NetFirewallHyperVRule` with WSL's
  VMCreatorId `{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}`).
- To start it at boot, a Task Scheduler task (trigger: at startup, run as that user with
  "Do not store password" / S4U) can run
  `wsl.exe -d Ubuntu-24.04 -- bash /path/to/start-avtr1.sh`.

On this PC, put the bridge URL in `.env.local` and restart the LiteAvatar server:

```sh
AVTR1_URL=http://<gpu-machine-ip>:8770   # or https://<tunnel-id>-8770.<region>.devtunnels.ms
# AVTR1_AVATAR=priya   # optional: another avatar id from the renderer's /avatars list
```

Then choose **Priti HD (AVTR-1, GPU server)**. The first selection renders a 6 s idle
clip on the GPU (a few seconds). Errors such as an unreachable bridge or an unknown
avatar id are shown under the text box. The bridge logs render speed per sentence;
below 1x real time the player buffers before speaking.
