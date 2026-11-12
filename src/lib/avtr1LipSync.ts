/**
 * Drives an AVTR-1 face (on the GPU machine, via /api/avtr1/render) from a live
 * audio track, e.g. a call's AI assistant (optionally pitch-shifted, e.g. to give a
 * man's face a deeper voice): the audio is cut into 0.6 s pieces, each
 * rendered in order (one session, so the motion flows on), and each piece is
 * played only once its frames are there, so voice and lips always match; frames
 * are drawn on the audio clock. Only speech is rendered: silent pieces show an idle
 * face loop instead, which keeps the GPU free for other calls (a call's AI usually
 * speaks well under half the time); while the voice runs late, silent pieces aren't
 * played either, which catches up. The whole call keeps one background frame and
 * only a soft-edged face patch changes, so lighting and grain stay perfectly steady.
 */

const CHUNK_S = 0.6; // a multiple of the bridge's 0.2 s render step; shorter = less delay, more requests
const MAX_LAG_S = 1.5; // beyond this delay (render + transfer), silent pieces are dropped to catch up
const SILENCE_RMS = 0.004; // quieter pieces count as silence
const IN_FLIGHT = 2; // renders sent ahead, so the network round trip overlaps the GPU work
const FPS = 25;
const BRIDGE_RATE = 16000;

// Posts blocks of samples from the audio thread, pitch-shifted by processorOptions.pitch
// (1 = unchanged): two read heads sweep a short delay line at (1 - pitch) samples per
// sample, cross-faded with sin^2 windows half a window apart (a classic delay-line shifter).
const WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.pitch = options.processorOptions?.pitch ?? 1;
    this.win = Math.round(sampleRate * 0.05);
    this.buf = new Float32Array(this.win * 4);
    this.w = 0;
    this.phase = 0;
  }
  read(d) {
    const n = this.buf.length, p = this.w - d, i = Math.floor(p), f = p - i;
    const a = this.buf[((i % n) + n) % n], b = this.buf[(((i + 1) % n) + n) % n];
    return a + (b - a) * f;
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    if (this.pitch === 1) {
      this.port.postMessage(ch.slice());
      return true;
    }
    const out = new Float32Array(ch.length), win = this.win, n = this.buf.length;
    for (let k = 0; k < ch.length; k++) {
      this.buf[this.w % n] = ch[k];
      this.phase = (this.phase + 1 - this.pitch + win) % win;
      const d1 = this.phase, d2 = (this.phase + win / 2) % win;
      const g1 = Math.sin(Math.PI * d1 / win) ** 2, g2 = Math.sin(Math.PI * d2 / win) ** 2;
      out[k] = g1 * this.read(d1 + 1) + g2 * this.read(d2 + 1);
      this.w++;
    }
    this.port.postMessage(out);
    return true;
  }
}
registerProcessor("tap", Tap);
`;

const IDLE_S = 4; // silence rendered once at the start: the background frame plus an idle face loop
const FEATHER_PX = 24; // soft edge where a face patch blends into the background

/** A face patch: drawn over the fixed background frame at x, y. */
type Patch = { img: ImageBitmap; x: number; y: number };

type Piece = {
  heardAt: number; // play-context time when the piece was complete
  start: number; // play-context time it plays from; -1 until scheduled
  audio: AudioBuffer;
  patches: Patch[];
  silent: boolean;
  skipped: boolean; // not rendered: the idle face loop shows instead
};

export class Avtr1LipSync {
  private capture: AudioContext | null = null;
  private play: AudioContext | null = null;
  private element: HTMLAudioElement | null = null;
  private pieces: Piece[] = []; // scheduled, in play order
  private nextAt = 0; // play-context time where the next piece can start
  private renders = Promise.resolve(); // one render at a time, in order
  private pending = 0; // renders queued or running
  private inFlight = 0;
  private waiting: (() => void)[] = []; // FIFO, so requests go out in piece order
  /** For a debug view: loudness of the latest piece, pieces heard / skipped / dropped, renders waiting, lag in s. */
  stats = { level: 0, pieces: 0, skipped: 0, dropped: 0, backlog: 0, lag: 0 };
  private session = crypto.randomUUID();
  private stopped = false;
  private timer = 0;
  // One background frame for the whole call, so lighting and grain never change; only the
  // face patch on top of it moves, from the idle loop or from speech.
  private base: ImageBitmap | null = null;
  private idle: Patch[] = [];
  private mask: HTMLCanvasElement | null = null;
  private scratch = document.createElement("canvas");

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly avatar: string,
    /** Whether the canvas shows the face (true) or the caller's placeholder should show (false). */
    private readonly onLive: (live: boolean) => void,
    private readonly onError: (message: string) => void,
    /** Voice pitch factor: 1 = as received, below 1 = deeper (0.8 is about 4 semitones down). */
    private readonly pitch = 1
  ) {}

  async start(track: MediaStreamTrack) {
    const stream = new MediaStream([track]);
    // Chrome only delivers a remote WebRTC track to Web Audio once a media element plays it.
    this.element = new Audio();
    this.element.srcObject = stream;
    this.element.muted = true;
    void this.element.play().catch(() => {});

    this.release();
    this.play = new AudioContext();
    this.capture = new AudioContext({ sampleRate: this.play.sampleRate });
    await this.capture.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" })));
    const source = this.capture.createMediaStreamSource(stream);
    const tap = new AudioWorkletNode(this.capture, "tap", { processorOptions: { pitch: this.pitch } });
    source.connect(tap);

    const rate = this.capture.sampleRate;
    const size = Math.round(rate * CHUNK_S);
    let buf = new Float32Array(size);
    let n = 0;
    tap.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
      for (let i = 0; i < data.length; i++) {
        buf[n++] = data[i];
        if (n === size) {
          this.addPiece(buf);
          buf = new Float32Array(size);
          n = 0;
        }
      }
    };
  }

  /** Stops following the voice (e.g. the participant left); the idle face keeps showing. */
  release() {
    void this.capture?.close();
    void this.play?.close();
    this.capture = this.play = null;
    if (this.element) this.element.srcObject = null;
    for (const p of this.pieces) p.patches.forEach((f) => f.img.close());
    this.pieces = [];
    this.nextAt = 0;
  }

  stop() {
    this.stopped = true;
    clearInterval(this.timer);
    this.release();
    this.idle.forEach((f) => f.img.close());
    this.base?.close();
  }

  /**
   * Renders a few seconds of silence once, before any voice: its first frame is the background
   * for the whole call, the rest the idle face loop. The face shows (onLive) once this is done.
   */
  async prepare() {
    const silence = new Int16Array(BRIDGE_RATE * IDLE_S);
    await this.render(silence, this.idle, { session: `${this.session}-idle`, key: true });
    if (this.stopped) return; // e.g. React's dev-mode remount
    if (!this.base || !this.idle.length) throw new Error("AVTR-1 sent no background frame.");
    this.mask = featherMask(this.idle[0].img.width, this.idle[0].img.height);
    // A timer rather than requestAnimationFrame, so it keeps going in a background tab.
    this.timer = window.setInterval(this.draw, 1000 / FPS);
  }

  private addPiece(samples: Float32Array<ArrayBuffer>) {
    if (this.stopped || !this.play) return;
    const play = this.play;
    const level = rms(samples);
    const piece: Piece = {
      heardAt: play.currentTime,
      start: -1,
      audio: play.createBuffer(1, samples.length, play.sampleRate),
      patches: [],
      silent: level < SILENCE_RMS,
      skipped: false,
    };
    // Silence shows the idle face loop; the GPU only renders speech.
    piece.skipped = piece.silent;
    piece.audio.copyToChannel(samples, 0);
    this.stats = { ...this.stats, level, pieces: this.stats.pieces + 1, skipped: this.stats.skipped + +piece.skipped, backlog: this.pending };

    const pcm = piece.skipped ? null : toPcm16k(samples, play.sampleRate);
    this.pending++;
    const rendered = pcm
      ? this.limited(() => this.render(pcm, piece.patches, { session: this.session, key: false })).catch((err) => {
          piece.skipped = true;
          this.onError(String(err.message ?? err));
        })
      : null;
    // Up to IN_FLIGHT renders run at once, but pieces are scheduled strictly in order.
    this.renders = this.renders
      .then(() => rendered)
      .then(() => this.schedule(piece))
      .finally(() => this.pending--);
  }

  private async limited(task: () => Promise<void>) {
    if (this.inFlight >= IN_FLIGHT) await new Promise<void>((go) => this.waiting.push(go));
    this.inFlight++;
    try {
      await task();
    } finally {
      this.inFlight--;
      this.waiting.shift()?.();
    }
  }

  /** Plays a piece right after the previous one, once its frames are there (pieces arrive in order). */
  private schedule(piece: Piece) {
    if (this.stopped || !this.play) return;
    const play = this.play;
    const start = Math.max(this.nextAt, play.currentTime + 0.05);
    const lag = start - piece.heardAt;
    this.stats.lag = lag;
    if (piece.silent && lag > MAX_LAG_S) {
      // Running late: drop a piece of silence to catch up.
      piece.patches.forEach((f) => f.img.close());
      this.stats.dropped++;
      return;
    }
    piece.start = start;
    this.nextAt = start + piece.audio.duration;
    this.pieces.push(piece);
    const node = play.createBufferSource();
    node.buffer = piece.audio;
    node.connect(play.destination);
    node.start(start);
  }

  /**
   * Renders int16 16 kHz audio on the GPU machine into face patches. With `key`, the first
   * frame comes whole and becomes the background (bridge protocol: 1 byte type + 4 byte
   * big-endian length + payload; 'K' full JPEG, 'P' x, y (2+2 bytes) + JPEG patch, 'E' end).
   */
  private async render(pcm: Int16Array, out: Patch[], { session, key }: { session: string; key: boolean }) {
    if (this.stopped) return;
    const query = `avatar=${encodeURIComponent(this.avatar)}&session=${session}&patch=1&key=${key ? 1 : 0}`;
    const res = await fetch(`/api/avtr1/render?${query}`, { method: "POST", body: pcm.buffer as ArrayBuffer });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => ({}))).error ?? `AVTR-1 render failed (${res.status})`);
    const reader = res.body.getReader();
    let buf = new Uint8Array(0);
    for (;;) {
      const { done, value } = await reader.read();
      if (done || this.stopped) return;
      const next = new Uint8Array(buf.length + value.length);
      next.set(buf);
      next.set(value, buf.length);
      buf = next;
      while (buf.length >= 5) {
        const len = new DataView(buf.buffer, buf.byteOffset + 1, 4).getUint32(0);
        if (buf.length < 5 + len) break;
        const kind = String.fromCharCode(buf[0]);
        const payload = buf.slice(5, 5 + len);
        buf = buf.slice(5 + len);
        if (kind === "K") {
          this.base ??= await createImageBitmap(new Blob([payload], { type: "image/jpeg" }));
        } else if (kind === "P") {
          const at = new DataView(payload.buffer, payload.byteOffset, 4);
          const img = await createImageBitmap(new Blob([payload.subarray(4)], { type: "image/jpeg" }));
          out.push({ img, x: at.getUint16(0), y: at.getUint16(2) });
        } else if (kind === "E") {
          const { error } = JSON.parse(new TextDecoder().decode(payload));
          if (error) throw new Error(error);
        }
      }
    }
  }

  private live = false;
  private setLive(live: boolean) {
    if (live === this.live) return;
    this.live = live;
    this.onLive(live);
  }

  private draw = () => {
    if (this.stopped || !this.base || !this.mask || !this.idle.length) return;
    if (this.play?.state === "suspended") void this.play.resume();
    const now = this.play?.currentTime ?? 0;
    // Drop pieces that have finished playing.
    while (this.pieces.length && this.pieces[0].start + this.pieces[0].audio.duration <= now) {
      this.pieces.shift()!.patches.forEach((f) => f.img.close());
    }
    const piece = this.pieces[0];
    let patch: Patch | undefined;
    if (piece && now >= piece.start && !piece.skipped) {
      patch = piece.patches[Math.min(piece.patches.length - 1, Math.floor((now - piece.start) * FPS))];
    }
    if (!patch) {
      // Idle: loop the silence render back and forth, so it never jumps at the loop point.
      const n = this.idle.length;
      const i = Math.floor((performance.now() / 1000) * FPS) % (2 * n - 2 || 1);
      patch = this.idle[i < n ? i : 2 * n - 2 - i];
    }
    if (this.canvas.width !== this.base.width) {
      this.canvas.width = this.base.width;
      this.canvas.height = this.base.height;
    }
    const ctx = this.canvas.getContext("2d")!;
    ctx.drawImage(this.base, 0, 0);
    // Feathered: the patch fades into the background over its outer edge, so no box shows.
    const s = this.scratch;
    if (s.width !== patch.img.width || s.height !== patch.img.height) {
      s.width = patch.img.width;
      s.height = patch.img.height;
    }
    const sc = s.getContext("2d")!;
    sc.globalCompositeOperation = "copy";
    sc.drawImage(patch.img, 0, 0);
    sc.globalCompositeOperation = "destination-in";
    sc.drawImage(this.mask, 0, 0);
    ctx.drawImage(s, patch.x, patch.y);
    this.setLive(true);
  };
}

/** Alpha mask for a w x h patch: opaque inside, fading to transparent over FEATHER_PX at the edges. */
function featherMask(w: number, h: number) {
  const mask = document.createElement("canvas");
  mask.width = w;
  mask.height = h;
  const ctx = mask.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const t = Math.min(1, Math.min(x, y, w - 1 - x, h - 1 - y) / FEATHER_PX);
      img.data[(y * w + x) * 4 + 3] = Math.round(255 * t * t * (3 - 2 * t)); // smoothstep
    }
  }
  ctx.putImageData(img, 0, 0);
  return mask;
}

function rms(samples: Float32Array) {
  let sum = 0;
  for (const s of samples) sum += s * s;
  return Math.sqrt(sum / samples.length);
}

/** Float32 samples at `rate` -> int16 at 16 kHz (linear interpolation). */
function toPcm16k(samples: Float32Array, rate: number) {
  const out = new Int16Array(Math.round((samples.length * BRIDGE_RATE) / rate));
  const step = rate / BRIDGE_RATE;
  for (let i = 0; i < out.length; i++) {
    const x = i * step;
    const j = Math.floor(x);
    const s = samples[j] + ((samples[j + 1] ?? samples[j]) - samples[j]) * (x - j);
    out[i] = Math.max(-1, Math.min(1, s)) * 0x7fff;
  }
  return out;
}
