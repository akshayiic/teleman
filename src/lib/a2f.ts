/** Browser side of the Audio2Face-3D pipeline: encode audio, fetch the track, play it back in sync. */

export type A2FTrack = {
  names: string[];
  frames: { t: number; w: number[] }[];
};

/** Blendshape weights keyed by lower-cased ARKit name, e.g. `jawopen`, `eyeblinkleft`. */
export type FaceWeights = Record<string, number>;

const SAMPLE_RATE = 16000;

/** Decodes any browser-supported audio and resamples it to 16 kHz mono PCM16. */
export async function encodeForA2F(data: ArrayBuffer, ctx: AudioContext) {
  const decoded = await ctx.decodeAudioData(data.slice(0));
  const length = Math.ceil(decoded.duration * SAMPLE_RATE);
  const offline = new OfflineAudioContext(1, length, SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded; // multi-channel input is down-mixed to the mono destination
  source.connect(offline.destination);
  source.start();
  const mono = (await offline.startRendering()).getChannelData(0);

  const pcm = new Int16Array(mono.length);
  for (let i = 0; i < mono.length; i++) {
    const s = Math.max(-1, Math.min(1, mono[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return { decoded, pcm };
}

export async function requestTrack(pcm: Int16Array): Promise<A2FTrack> {
  const res = await fetch("/api/a2f", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: pcm.buffer as ArrayBuffer,
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error ?? `Audio2Face request failed (${res.status}).`);
  return body;
}

/**
 * Plays decoded audio and exposes the blendshape weights for the current
 * playback time, interpolated between the ~30 fps frames.
 */
export class A2FPlayer {
  private source: AudioBufferSourceNode | null = null;
  private startedAt = 0;
  private track: A2FTrack | null = null;
  private keys: string[] = [];
  private weights: FaceWeights = {};

  constructor(private ctx: AudioContext) {}

  play(audio: AudioBuffer, track: A2FTrack, onEnd: () => void) {
    this.stop();
    this.track = track;
    this.keys = track.names.map((n) => n.toLowerCase());
    const source = this.ctx.createBufferSource();
    source.buffer = audio;
    source.connect(this.ctx.destination);
    source.onended = () => {
      if (this.source === source) {
        this.source = null;
        this.track = null;
        onEnd();
      }
    };
    this.startedAt = this.ctx.currentTime;
    source.start();
    this.source = source;
  }

  stop() {
    const source = this.source;
    this.source = null;
    this.track = null;
    source?.stop();
  }

  /** Current weights, or null when nothing is playing. */
  sample(): FaceWeights | null {
    const track = this.track;
    if (!track || track.frames.length === 0) return null;
    const t = this.ctx.currentTime - this.startedAt;
    const frames = track.frames;

    // Binary search for the frame at or before t.
    let lo = 0;
    let hi = frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (frames[mid].t <= t) lo = mid;
      else hi = mid - 1;
    }
    const a = frames[lo];
    const b = frames[Math.min(lo + 1, frames.length - 1)];
    const k = b.t > a.t ? Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t))) : 0;
    for (let i = 0; i < this.keys.length; i++) {
      this.weights[this.keys[i]] = a.w[i] + (b.w[i] - a.w[i]) * k;
    }
    return this.weights;
  }
}
