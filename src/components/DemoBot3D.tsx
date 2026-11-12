"use client";

import { useEffect, useRef, useState } from "react";
import { VConfCall } from "@/lib/vconf";

/**
 * /3d-bot: a 3D Avaturn avatar (public/avatars/avaturn.glb, with the 15 Oculus visemes
 * and ARKit face shapes), rendered by TalkingHead in this browser; no GPU server.
 * Two ways to drive the lips:
 * - Text (the box below, or ?text=): HeadTTS (Kokoro, in the browser) speaks it and
 *   returns each phoneme's timing, so every mouth shape lands on its sound.
 * - Audio (a call's AI assistant, or ?audio=<sound file url>): HeadAudio classifies
 *   the sound into visemes live (no transcript needed, but it's an estimate).
 * The call is the same V-Retail conference as /demo-bot: open with ?jwt=<customer
 * token>, one click starts the camera and mic and joins; backend messages are logged
 * to the browser console. Add &debug for the participants and the voice's level.
 */

const SERVER = process.env.NEXT_PUBLIC_VCONF_SERVER ?? "https://backend-1.conference.svc.inkincaps.com";
const ASSISTANT = /assistant|\bai\b|bot/i; // display name of the backend's AI participant
const AVATAR = { url: "/avatars/avaturn.glb", body: "F" };
/** Kokoro voices (huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX#voices). */
const VOICES = {
  af_heart: "Heart (US)",
  af_bella: "Bella (US)",
  hf_alpha: "Alpha (Indian)",
  hf_beta: "Beta (Indian)",
  bf_emma: "Emma (UK)",
};
const AUDIO_DELAY_S = 0.08; // the voice is delayed this much so HeadAudio's lips keep up

/* eslint-disable @typescript-eslint/no-explicit-any -- TalkingHead, HeadTTS and HeadAudio are untyped JS */
type Head = any;
type HeadAudioNode = any;
type Tts = any;

export default function DemoBot3D() {
  const [{ jwt, audio, debug, initialText }] = useState(() => {
    const q = new URLSearchParams(location.search); // browser-only component
    return { jwt: q.get("jwt"), audio: q.get("audio"), debug: q.has("debug"), initialText: q.get("text") };
  });
  const [text, setText] = useState(initialText ?? "Hello! I am your shopping assistant. How can I help you today?");
  const [voice, setVoice] = useState<keyof typeof VOICES>("af_heart");
  const [statuses, setStatuses] = useState<Record<string, string | null>>({});
  const status = Object.values(statuses).filter(Boolean).join(" · ") || null;
  const setStatus = (source: string, value: string | null) => setStatuses((s) => ({ ...s, [source]: value }));
  const [started, setStarted] = useState(false);
  const [debugText, setDebugText] = useState("");

  const container = useRef<HTMLDivElement>(null);
  const self = useRef<HTMLVideoElement>(null);
  const head = useRef<Head>(null);
  const headAudio = useRef<HeadAudioNode>(null);
  const tts = useRef<Tts>(null);
  const route = useRef<{ source: AudioNode; delay: DelayNode; element?: HTMLAudioElement } | null>(null);
  const call = useRef<VConfCall | null>(null);
  const following = useRef<string | null>(null); // the participant whose voice drives the avatar
  const players = useRef(new Map<string, HTMLAudioElement>()); // other people's voices
  const peers = useRef(new Map<string, string>());
  const camera = useRef<MediaStream | null>(null);
  const audioPeers = useRef(new Set<string>());
  const vad = useRef({ db: -100, active: false });

  // Avatar, plus HeadAudio for audio-driven lip-sync.
  useEffect(() => {
    let cancelled = false;
    let instance: Head = null;
    (async () => {
      setStatus("avatar", "Loading avatar…");
      const [{ TalkingHead }, { LipsyncEn }, { HeadAudio }] = await Promise.all([
        import("@met4citizen/talkinghead"),
        import("@met4citizen/talkinghead/modules/lipsync-en.mjs"),
        import("@met4citizen/headaudio/modules/headaudio.mjs"),
      ]);
      if (cancelled || !container.current) return;
      instance = new TalkingHead(container.current, {
        ttsEndpoint: null, // speech comes from HeadTTS or the call, not Google TTS
        lipsyncModules: [], // its dynamic import() of these doesn't survive bundling…
        lipsyncLang: "en",
        cameraView: "upper",
        avatarMood: "neutral",
      });
      instance.lipsync.en = new LipsyncEn(); // …so register English lip-sync directly.
      await instance.showAvatar(
        { url: AVATAR.url, body: AVATAR.body, avatarMood: "neutral", lipsyncLang: "en" },
        (ev: ProgressEvent) => {
          if (ev.lengthComputable) setStatus("avatar", `Loading avatar… ${Math.round((ev.loaded / ev.total) * 100)}%`);
        }
      );
      if (cancelled) return;

      await instance.audioCtx.audioWorklet.addModule("/headaudio/headworklet.min.mjs");
      const node = new HeadAudio(instance.audioCtx, {
        processorOptions: { vadEventsEnabled: true },
        // speakerMeanHz: the avatar is a woman, so the assistant voice is expected to be one too.
        parameterData: { vadGateActiveDb: -40, vadGateInactiveDb: -60, silMode: 0, speakerMeanHz: 220 },
      });
      await node.loadModel("/headaudio/model-en-mixed.bin");
      if (cancelled) return;
      node.onvalue = (key: string, value: number) => {
        Object.assign(instance.mtAvatar[key], { newvalue: value, needsUpdate: true });
      };
      node.onstarted = () => instance.lookAtCamera(500);
      node.onvad = (d: { db: number; active: number }) => (vad.current = { db: d.db, active: d.active > 0 });
      instance.opt.update = node.update.bind(node); // runs in TalkingHead's animation loop
      head.current = instance;
      headAudio.current = node;
      setStatus("avatar", null);
    })().catch((err) => {
      if (!cancelled) setStatus("avatar", `Avatar failed to load: ${err?.message ?? err}`);
    });
    return () => {
      cancelled = true;
      head.current = null;
      headAudio.current = null;
      instance?.stop?.();
      instance?.dispose?.();
    };
  }, []);

  // HeadTTS (Kokoro voice + phoneme timings) for text, loaded once.
  useEffect(() => {
    let cancelled = false;
    let client: Tts = null;
    (async () => {
      const origin = window.location.origin;
      const { HeadTTS } = await import(
        /* webpackIgnore: true */ /* turbopackIgnore: true */ `${origin}/headtts/modules/headtts.mjs`
      );
      if (cancelled) return;
      client = new HeadTTS({
        endpoints: ["webgpu", "wasm"],
        transformersModule: `${origin}/headtts/vendor/transformers.min.js`,
        languages: ["en-us"],
        voices: Object.keys(VOICES),
        defaultVoice: "af_heart",
        dtypeWasm: "q8",
      });
      client.onmessage = (message: { type: string; data: any }) => {
        if (message.type === "audio") head.current?.speakAudio(message.data, {}, null);
        else if (message.type === "error") setStatus("voice", `Speech failed: ${message.data?.error ?? "unknown error"}`);
      };
      setStatus("voice", "Loading voice model (first time only, ~100–300 MB)…");
      await client.connect(null, (ev: ProgressEvent) => {
        if (ev.lengthComputable && !cancelled) setStatus("voice", `Loading voice model… ${Math.round((ev.loaded / ev.total) * 100)}%`);
      });
      if (cancelled) return;
      tts.current = client;
      setStatus("voice", null);
    })().catch((err) => {
      if (!cancelled) setStatus("voice", `Voice failed to load: ${err?.message ?? err}`);
    });
    return () => {
      cancelled = true;
      tts.current = null;
      client?.ww?.terminate(); // HeadTTS has no close()
    };
  }, []);

  useEffect(() => {
    if (!debug) return;
    const timer = setInterval(() => {
      const lines = [...peers.current].map(
        ([id, name]) =>
          `${name || id}${audioPeers.current.has(id) ? " · audio" : ""}${following.current === id ? " · drives the avatar" : ""}`
      );
      if (route.current) lines.push(`voice ${vad.current.db.toFixed(0)} dB · ${vad.current.active ? "speech" : "silence"} (speech above -40 dB)`);
      setDebugText(lines.join("\n") || "no other participants");
    }, 300);
    return () => clearInterval(timer);
  }, [debug]);

  /** Speaks text with exact, phoneme-timed lip-sync. */
  const speak = () => {
    if (!text.trim()) return;
    if (!head.current || !tts.current) return setStatus("voice", "Still loading, try again in a moment…");
    void head.current.audioCtx.resume();
    tts.current.setup({ voice, language: "en-us", speed: 1, audioEncoding: "wav" });
    tts.current.synthesize({ input: text });
  };

  const stopSpeaking = () => head.current?.stopSpeaking();

  /** Audio-driven lip-sync: `source` goes to HeadAudio, and (slightly delayed) to the speakers. */
  const follow = (makeSource: (ctx: AudioContext) => AudioNode, element?: HTMLAudioElement) => {
    const h = head.current, node = headAudio.current;
    if (!h || !node) return setStatus("call", "The avatar is still loading.");
    unfollow();
    const ctx: AudioContext = h.audioCtx;
    void ctx.resume();
    const source = makeSource(ctx);
    const delay = new DelayNode(ctx, { delayTime: AUDIO_DELAY_S });
    source.connect(node);
    source.connect(delay).connect(ctx.destination);
    route.current = { source, delay, element };
  };

  const unfollow = () => {
    const r = route.current;
    if (!r) return;
    route.current = null;
    if (r.source instanceof AudioBufferSourceNode) r.source.stop();
    r.source.disconnect();
    r.delay.disconnect();
    if (r.element) r.element.srcObject = null;
  };

  const followTrack = (track: MediaStreamTrack) => {
    const stream = new MediaStream([track]);
    // Chrome only delivers a remote WebRTC track to Web Audio once a media element plays it.
    const element = new Audio();
    element.srcObject = stream;
    element.muted = true;
    void element.play().catch(() => {});
    follow((ctx) => ctx.createMediaStreamSource(stream), element);
  };

  const stopAll = () => {
    call.current?.close();
    call.current = null;
    unfollow();
    following.current = null;
    for (const p of players.current.values()) p.srcObject = null;
    players.current.clear();
    camera.current?.getTracks().forEach((t) => t.stop());
    camera.current = null;
  };
  const stopAllRef = useRef(stopAll);
  useEffect(() => {
    stopAllRef.current = stopAll;
  });
  useEffect(() => () => stopAllRef.current(), []);

  const onRemoteTrack = (peerId: string, kind: "audio" | "video", track: MediaStreamTrack | null) => {
    if (kind !== "audio") return; // the assistant has no video; others' isn't shown here
    if (track) audioPeers.current.add(peerId);
    else audioPeers.current.delete(peerId);
    if (!track) {
      if (following.current === peerId) {
        unfollow();
        following.current = null;
      }
      const player = players.current.get(peerId);
      if (player) player.srcObject = null;
      players.current.delete(peerId);
      return;
    }
    const name = peers.current.get(peerId) ?? "";
    console.log("[3d-bot] audio from", name || peerId);
    if (!following.current && (ASSISTANT.test(name) || peers.current.size <= 1)) {
      following.current = peerId;
      followTrack(track);
      return;
    }
    const player = new Audio();
    player.srcObject = new MediaStream([track]);
    void player.play().catch(() => {});
    players.current.set(peerId, player);
  };

  /** ?audio=: the avatar follows a sound file, no call. */
  const playFile = async () => {
    const h = head.current;
    if (!h) return setStatus("call", "The avatar is still loading.");
    setStarted(true);
    try {
      // Decoded and played in the avatar's audio context, so sound and lips share one clock.
      const ctx: AudioContext = h.audioCtx;
      const buffer = await ctx.decodeAudioData(await (await fetch(audio!)).arrayBuffer());
      following.current = "file";
      follow(() => {
        const source = new AudioBufferSourceNode(ctx, { buffer });
        source.onended = () => {
          if (route.current?.source !== source) return;
          unfollow();
          following.current = null;
          setStarted(false);
        };
        source.start();
        return source;
      });
    } catch (err) {
      setStarted(false);
      setStatus("call", `Can't play ${audio}: ${err instanceof Error ? err.message : err}`);
    }
  };

  const start = async () => {
    if (audio) return playFile();
    if (!jwt) return;
    setStarted(true);
    setStatus("call", "Starting camera and microphone…");
    let media: MediaStream;
    try {
      media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true }, video: true });
    } catch {
      media = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true } }).catch(() => new MediaStream());
    }
    camera.current = media;
    if (self.current) self.current.srcObject = media;

    const c = new VConfCall(
      SERVER,
      jwt,
      null,
      {
        onStatus: (s) => setStatus("call", s),
        onPeers: (p) => {
          peers.current = p;
          console.log("[3d-bot] in the call with", [...p.values()]);
        },
        onNotification: (method, data) => console.log("[3d-bot] backend:", method, data),
        onRemoteTrack,
        onClosed: (reason) => {
          stopAll();
          setStarted(false);
          setStatus("call", reason === "Left the call." ? null : reason);
        },
      },
      { receive: true }
    );
    call.current = c;
    await c.join(media).catch(() => {}); // onClosed reports why
  };

  return (
    <div className="stage">
      <div ref={container} className="talking-avatar" />
      <div className="credit">Avatar made with Avaturn · lip-sync: HeadTTS (text) / HeadAudio (audio)</div>
      <video ref={self} className="demo-bot-self" autoPlay muted playsInline hidden={!started || !!audio} />
      {debug && <pre className="demo-bot-debug">{debugText}</pre>}

      <div className="panel">
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} placeholder="Type something to say…" />
        <div className="row">
          <button onClick={speak}>Speak</button>
          <button onClick={stopSpeaking}>Stop</button>
          <select value={voice} onChange={(e) => setVoice(e.target.value as keyof typeof VOICES)}>
            {Object.entries(VOICES).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
          {(jwt || audio) &&
            (started ? (
              <button onClick={() => (stopAll(), setStarted(false))}>{audio ? "Stop audio" : "Leave call"}</button>
            ) : (
              <button onClick={start}>{audio ? "Play audio" : "Join call"}</button>
            ))}
        </div>
        {status && <div className="status">{status}</div>}
      </div>
    </div>
  );
}
