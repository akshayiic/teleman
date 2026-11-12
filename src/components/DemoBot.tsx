"use client";

import { useEffect, useRef, useState } from "react";
import { Avtr1LipSync } from "@/lib/avtr1LipSync";
import { VConfCall } from "@/lib/vconf";

/**
 * /demo-bot: a V-Retail conference call from the customer's side, where the
 * backend's AI assistant (an audio-only participant) is shown as Geeta: its voice
 * drives his face live on the AVTR-1 GPU machine. Open with ?jwt=<customer token>;
 * one click starts the camera and mic (browsers need it for sound) and joins.
 * Backend messages are logged to the browser console.
 * Without a call, ?audio=<url of a sound file> drives Geeta from that file instead.
 * Add &debug for an overlay with the participants and the followed voice's level.
 */

const SERVER = process.env.NEXT_PUBLIC_VCONF_SERVER ?? "https://backend-1.conference.svc.inkincaps.com";
const AVATAR = "geeta"; // AVTR-1 avatar id (PNG name in its reference_frames)
const ASSISTANT = /assistant|\bai\b|bot/i; // display name of the backend's AI participant
const VOICE_PITCH = 1; // the backend's own voice; below 1 deepens it (0.8 suited a man's face)

export default function DemoBot() {
  const [{ jwt, audio, debug }] = useState(() => {
    const q = new URLSearchParams(location.search); // browser-only component
    return { jwt: q.get("jwt"), audio: q.get("audio"), debug: q.has("debug") };
  });
  const [started, setStarted] = useState(false);
  const [live, setLive] = useState(false); // the rendered face is ready (else her photo shows)
  const [status, setStatus] = useState<string | null>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const self = useRef<HTMLVideoElement>(null);
  const call = useRef<VConfCall | null>(null);
  const lipSync = useRef<Avtr1LipSync | null>(null); // one per page: fixed background + idle face from the start
  const following = useRef<string | null>(null); // the participant whose voice drives Geeta
  const players = useRef(new Map<string, HTMLAudioElement>()); // other people's voices
  const peers = useRef(new Map<string, string>());
  const camera = useRef<MediaStream | null>(null);
  const audioPeers = useRef(new Set<string>());
  const [debugText, setDebugText] = useState("");

  useEffect(() => {
    if (!debug) return;
    const timer = setInterval(() => {
      const lines = [...peers.current].map(
        ([id, name]) =>
          `${name || id}${audioPeers.current.has(id) ? " · audio" : ""}${following.current === id ? " · drives Geeta" : ""}`
      );
      const s = following.current ? lipSync.current?.stats : null;
      if (s) {
        lines.push(`voice level ${(s.level * 1000).toFixed(1)} (speech > 4) · ${s.pieces} pieces heard`);
        lines.push(`delay ${s.lag.toFixed(1)} s · backlog ${s.backlog} · ${s.skipped} skipped · ${s.dropped} dropped`);
      }
      setDebugText(lines.join("\n") || "no other participants");
    }, 500);
    return () => clearInterval(timer);
  }, [debug]);

  const stopAll = () => {
    call.current?.close();
    call.current = null;
    lipSync.current?.release();
    following.current = null;
    for (const p of players.current.values()) p.srcObject = null;
    players.current.clear();
    camera.current?.getTracks().forEach((t) => t.stop());
    camera.current = null;
  };

  // Render the background and idle face as soon as the page opens, so nothing changes later.
  useEffect(() => {
    const sync = new Avtr1LipSync(canvas.current!, AVATAR, setLive, setStatus, VOICE_PITCH);
    lipSync.current = sync;
    void sync.prepare().catch((err) => setStatus(String(err.message ?? err)));
    return () => {
      stopAll();
      sync.stop();
    };
  }, []);

  const onRemoteTrack = (peerId: string, kind: "audio" | "video", track: MediaStreamTrack | null) => {
    if (kind !== "audio") return; // the assistant has no video; others' isn't shown here
    if (track) audioPeers.current.add(peerId);
    else audioPeers.current.delete(peerId);
    if (!track) {
      if (following.current === peerId) {
        lipSync.current?.release();
        following.current = null;
      }
      players.current.get(peerId)?.remove();
      players.current.delete(peerId);
      return;
    }
    const name = peers.current.get(peerId) ?? "";
    console.log("[demo-bot] audio from", name || peerId);
    if (!following.current && (ASSISTANT.test(name) || peers.current.size <= 1)) {
      following.current = peerId;
      void lipSync.current?.start(track);
      return;
    }
    const player = new Audio();
    player.srcObject = new MediaStream([track]);
    void player.play().catch(() => {});
    players.current.set(peerId, player);
  };

  /** ?audio=: Geeta follows a sound file, no call. */
  const playFile = async () => {
    setStarted(true);
    const player = new Audio(audio!);
    player.crossOrigin = "anonymous";
    await player.play().catch((err) => setStatus(String(err)));
    const [track] = (player as HTMLAudioElement & { captureStream(): MediaStream }).captureStream().getAudioTracks();
    player.muted = true; // heard through the lip sync, in step with the face
    player.onended = () => setStatus("End of the audio file.");
    following.current = "file";
    void lipSync.current?.start(track);
  };

  const start = async () => {
    if (audio) return playFile();
    if (!jwt) return;
    setStarted(true);
    setStatus("Starting camera and microphone…");
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
        onStatus: setStatus,
        onPeers: (p) => {
          peers.current = p;
          console.log("[demo-bot] in the call with", [...p.values()]);
        },
        onNotification: (method, data) => console.log("[demo-bot] backend:", method, data),
        onRemoteTrack,
        onClosed: (reason) => {
          stopAll();
          setStarted(false);
          setStatus(reason === "Left the call." ? null : reason);
        },
      },
      { receive: true }
    );
    call.current = c;
    await c.join(media).catch(() => {}); // onClosed reports why
  };

  return (
    <div className="demo-bot">
      <div className="tavus">
        {/* Her photo for the few seconds until the rendered face is ready (a placeholder, so no next/image). */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="liteavatar-media" src={`/avatars/${AVATAR}.png`} alt="" hidden={live} />
        <canvas ref={canvas} className="liteavatar-media" hidden={!live} />
      </div>
      <video ref={self} className="demo-bot-self" autoPlay muted playsInline hidden={!started || !!audio} />
      {(jwt || audio) && !started && (
        <button className="demo-bot-start" onClick={start}>
          Start
        </button>
      )}
      {debug && <pre className="demo-bot-debug">{debugText}</pre>}
      {(!jwt && !audio) || status ? (
        <div className="demo-bot-status">{status ?? "Open this page with ?jwt=<token> to join a call."}</div>
      ) : null}
    </div>
  );
}
