"use client";

import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import type { TalkingAvatarHandle } from "./TalkingAvatar";

/**
 * Free, unlimited, photoreal presenter rendered on this machine by the local
 * LiteAvatar server (liteavatar-server/server.py, reached via the
 * /liteavatar rewrite). Idle: the avatar's own clip, mouth closed. Speaking:
 * the server streams audio plus lip-synced frames per segment; we buffer
 * enough frames to (usually) never stall, then draw them in step with the
 * audio clock. If rendering falls behind (e.g. a throttled CPU), the audio is
 * paused until the next frame arrives, so lips and voice never drift apart.
 */

type LiteAvatarPresenterProps = {
  ref?: Ref<TalkingAvatarHandle>;
  /** Face name known to the server (AVATARS in liteavatar-server/server.py). */
  avatar: string;
  /** Voice language; "auto" lets the server pick Hindi for Devanagari text. */
  lang: "auto" | "en" | "hi";
  onStatus(source: "liteavatar", status: string | null): void;
  onSpeakingChange(speaking: boolean): void;
};

type Segment = {
  fps: number;
  total: number;
  startFrame: number;
  audio: Promise<AudioBuffer> | null;
  frames: ImageBitmap[];
  /** Wall-clock ms when the first frame of this segment arrived, for rate estimates. */
  firstFrameAt: number;
};

const SOURCE_FPS = 30; // fps of the avatar's idle clip
const OUTPUT_FPS = 25; // frame rate of outputStream()'s video
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Position in the idle clip matching the server's ping-pong frame order. */
function idleTime(frame: number, clipFrames: number) {
  const i = Math.floor(frame) % (2 * clipFrames);
  return (i < clipFrames ? i : 2 * clipFrames - 1 - i) / SOURCE_FPS;
}

export default function LiteAvatarPresenter({ ref, avatar, lang, onStatus, onSpeakingChange }: LiteAvatarPresenterProps) {
  const [showFrames, setShowFrames] = useState(false); // canvas over the idle clip while speaking
  const idle = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const audioCtx = useRef<AudioContext | null>(null);
  const session = useRef(0); // bumped to cancel the current speech
  const abort = useRef<AbortController | null>(null);
  const source = useRef<AudioBufferSourceNode | null>(null);
  const showingFrames = useRef(false); // showFrames, readable from the output loop
  const output = useRef<{ stream: MediaStream; audio: MediaStreamAudioDestinationNode; timer: number } | null>(null);
  const callbacks = useRef({ onStatus, onSpeakingChange });
  useEffect(() => {
    callbacks.current = { onStatus, onSpeakingChange };
  });

  useEffect(() => {
    let cancelled = false;
    // Also tells the server to start loading this face.
    fetch(`/liteavatar/health?avatar=${encodeURIComponent(avatar)}`)
      .then(async (res) => {
        // A JSON error comes from the server itself (e.g. the AVTR-1 bridge is unreachable).
        const body = await res.json().catch(() => null);
        if (!res.ok) throw new Error(body?.error ?? "");
        if (!cancelled) callbacks.current.onStatus("liteavatar", null);
      })
      .catch((err) => {
        if (!cancelled) {
          callbacks.current.onStatus(
            "liteavatar",
            err?.message || "The local LiteAvatar server isn't running. Start it with: npm run liteavatar"
          );
        }
      });
    return () => {
      cancelled = true;
      stop();
      if (output.current) {
        clearInterval(output.current.timer);
        output.current.stream.getTracks().forEach((t) => t.stop());
        output.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setTalking = (on: boolean) => {
    if (!on) {
      setShowFrames(false);
      showingFrames.current = false;
    }
    callbacks.current.onSpeakingChange(on);
  };

  /**
   * One canvas showing whatever is on screen (idle clip or speech frames) plus the
   * speech audio, as a MediaStream. Drawn on a timer rather than requestAnimationFrame
   * so it keeps going (if slower) when the tab is in the background.
   */
  const outputStream = () => {
    if (output.current) return output.current.stream;
    const ctx = (audioCtx.current ??= new AudioContext());
    void ctx.resume();
    const out = document.createElement("canvas");
    out.width = 1280;
    out.height = 720;
    const draw = out.getContext("2d")!;
    const timer = window.setInterval(() => {
      const src = showingFrames.current ? canvas.current : idle.current;
      if (!src) return;
      const w = src instanceof HTMLVideoElement ? src.videoWidth : src.width;
      const h = src instanceof HTMLVideoElement ? src.videoHeight : src.height;
      if (!w || !h) return;
      if (out.width !== w || out.height !== h) {
        out.width = w;
        out.height = h;
      }
      draw.drawImage(src, 0, 0);
    }, 1000 / OUTPUT_FPS);
    const audio = ctx.createMediaStreamDestination();
    const stream = new MediaStream([...out.captureStream(OUTPUT_FPS).getVideoTracks(), ...audio.stream.getAudioTracks()]);
    output.current = { stream, audio, timer };
    return stream;
  };

  const stop = () => {
    session.current++;
    abort.current?.abort();
    abort.current = null;
    try {
      source.current?.stop();
    } catch {}
    source.current = null;
    void audioCtx.current?.resume(); // may have been held for a late frame
    setTalking(false);
  };

  /** Reads the server's binary stream into segments, calling back as data arrives. */
  const readStream = async (res: Response, segments: Segment[], ctx: AudioContext, isCurrent: () => boolean) => {
    const reader = res.body!.getReader();
    let buf = new Uint8Array(0);
    for (;;) {
      const { done, value } = await reader.read();
      if (done || !isCurrent()) return;
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
        const seg = segments[segments.length - 1];
        if (kind === "S") {
          const h = JSON.parse(new TextDecoder().decode(payload));
          segments.push({ fps: h.fps, total: h.frames, startFrame: h.startFrame, audio: null, frames: [], firstFrameAt: 0 });
        } else if (kind === "A" && seg) {
          seg.audio = ctx.decodeAudioData(payload.buffer);
        } else if (kind === "F" && seg) {
          if (!seg.firstFrameAt) seg.firstFrameAt = performance.now();
          seg.frames.push(await createImageBitmap(new Blob([payload], { type: "image/jpeg" })));
        } else if (kind === "E") {
          const { error } = JSON.parse(new TextDecoder().decode(payload));
          if (error) throw new Error(error);
        }
      }
    }
  };

  /** Plays one segment: waits until playback can't outrun rendering, then draws frames on the audio clock. */
  const playSegment = async (seg: Segment, ctx: AudioContext, isCurrent: () => boolean) => {
    const ready = () => {
      const have = seg.frames.length;
      if (have >= seg.total) return true;
      if (have < 10) return false;
      // Frames still to render, at the rate they've been arriving, must finish
      // before the audio needs them (worst case: the last frame), with margin.
      const rate = (have - 1) / ((performance.now() - seg.firstFrameAt) / 1000);
      return ((seg.total - have) / rate) * 1.3 <= seg.total / seg.fps;
    };
    while (!ready() || !seg.audio) {
      if (!isCurrent()) return;
      await sleep(30);
    }
    const audio = await seg.audio;
    if (!isCurrent()) return;

    const node = ctx.createBufferSource();
    node.buffer = audio;
    node.connect(ctx.destination);
    if (output.current) node.connect(output.current.audio);
    const startAt = ctx.currentTime + 0.05;
    node.start(startAt);
    source.current = node;
    const draw = canvas.current!.getContext("2d")!;
    const duration = Math.max(audio.duration, seg.total / seg.fps);

    await new Promise<void>((resolve) => {
      const tick = () => {
        if (!isCurrent()) return resolve();
        const t = ctx.currentTime - startAt;
        const want = Math.max(0, Math.floor(t * seg.fps));
        // Rendering fell behind: hold the audio until the frame exists.
        if (want >= seg.frames.length && seg.frames.length < seg.total) {
          if (ctx.state === "running") void ctx.suspend();
          return requestAnimationFrame(tick);
        }
        if (ctx.state === "suspended") void ctx.resume();
        const i = Math.min(seg.frames.length - 1, want);
        const frame = seg.frames[i];
        if (frame) {
          if (canvas.current!.width !== frame.width) {
            canvas.current!.width = frame.width;
            canvas.current!.height = frame.height;
          }
          draw.drawImage(frame, 0, 0);
          setShowFrames(true);
          showingFrames.current = true;
        }
        if (t >= duration) return resolve();
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    seg.frames.forEach((f) => f.close());
  };

  const speak = async (text: string) => {
    stop();
    const mine = session.current;
    const isCurrent = () => session.current === mine;
    // Created/resumed inside the click so sound is allowed.
    const ctx = (audioCtx.current ??= new AudioContext());
    void ctx.resume();
    const controller = new AbortController();
    abort.current = controller;
    setTalking(true);
    callbacks.current.onStatus("liteavatar", "Preparing speech…");

    // Continue from where the idle clip is, so the body doesn't jump.
    const startFrame = Math.floor((idle.current?.currentTime ?? 0) * SOURCE_FPS);
    const segments: Segment[] = [];
    try {
      const res = await fetch("/liteavatar/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, startFrame, avatar, lang: lang === "auto" ? null : lang }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `LiteAvatar server error (${res.status}). Is it running? npm run liteavatar`);
      }
      let streamDone = false;
      let failed: unknown = null;
      readStream(res, segments, ctx, isCurrent).then(
        () => (streamDone = true),
        (err) => {
          failed = err;
          streamDone = true;
        }
      );

      for (let n = 0; ; n++) {
        while (!segments[n] && !streamDone) {
          if (!isCurrent()) return;
          await sleep(30);
        }
        if (failed) throw failed;
        const seg = segments[n];
        if (!seg || !isCurrent()) break;
        if (n === 0) callbacks.current.onStatus("liteavatar", null);
        // Stop waiting for frames that will never come if the stream broke mid-segment.
        await playSegment(seg, ctx, () => isCurrent() && !failed);
        if (failed) throw failed;
        // Resume the idle clip from the body position the speech ended on.
        if (idle.current && isCurrent()) {
          idle.current.currentTime = idleTime(seg.startFrame + (seg.total * SOURCE_FPS) / seg.fps, 150);
        }
      }
    } catch (err) {
      if (!isCurrent() || (err instanceof DOMException && err.name === "AbortError")) return;
      callbacks.current.onStatus("liteavatar", err instanceof Error ? err.message : String(err));
    } finally {
      if (isCurrent()) {
        abort.current = null;
        setTalking(false);
      }
    }
  };

  useImperativeHandle(ref, () => ({ speak: (text: string) => void speak(text), stop, outputStream }));

  return (
    <div className="tavus">
      <video
        ref={idle}
        className="liteavatar-media"
        src={`/liteavatar/idle.webm?avatar=${encodeURIComponent(avatar)}`}
        autoPlay
        muted
        loop
        playsInline
        hidden={showFrames}
      />
      <canvas ref={canvas} className="liteavatar-media" hidden={!showFrames} />
    </div>
  );
}
