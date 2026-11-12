"use client";

import { useEffect, useRef, useState } from "react";
import { Avtr1LipSync } from "@/lib/avtr1LipSync";
import { chooseMic, micMeter, microphones, openMic, saveMicId } from "@/lib/microphone";
import { VConfCall } from "@/lib/vconf";

/**
 * /demo-bot: a V-Retail conference call from the customer's side, where the
 * backend's AI assistant (an audio-only participant) is shown as Geeta: its voice
 * drives her face live on the AVTR-1 GPU machine. Open with ?jwt= (or ?t=) <customer token>;
 * one click starts the camera and mic (browsers need it for sound) and joins. The mic is
 * chosen exactly (see lib/microphone) and can be switched in the corner, with a level bar,
 * since the AI only answers once it hears the customer.
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
    // ?jwt= as in the conference client's harness, or ?t= as in "instant meeting" links
    return { jwt: q.get("jwt") ?? q.get("t"), audio: q.get("audio"), debug: q.has("debug") };
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
  const tracks = useRef(new Map<string, MediaStreamTrack>()); // each participant's current audio track
  const followedTrack = useRef<MediaStreamTrack | null>(null); // the track driving Geeta
  const [debugText, setDebugText] = useState("");
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [micId, setMicId] = useState("");
  const [micLevel, setMicLevel] = useState(0);
  const micLevelRef = useRef(0);
  const stopMeter = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!debug) return;
    let lastBytes = 0;
    const timer = setInterval(async () => {
      const lines = [...peers.current].map(
        ([id, name]) =>
          `${name || id}${audioPeers.current.has(id) ? " · audio" : ""}${following.current === id ? " · drives Geeta" : ""}`
      );
      const micTrack = camera.current?.getAudioTracks()[0];
      if (micTrack) lines.push(`your mic: ${micTrack.label} · level ${(micLevelRef.current * 1000).toFixed(1)}`);
      const s = following.current ? lipSync.current?.stats : null;
      if (s) {
        lines.push(`voice level ${(s.level * 1000).toFixed(1)} (speech > 4) · ${s.pieces} pieces heard`);
        lines.push(
          `delay ${s.lag.toFixed(1)} s · render ${s.renderS.toFixed(2)} s · backlog ${s.backlog} · ${s.late} late · ${s.skipped} silent · ${s.dropped} dropped`
        );
      }
      // What WebRTC itself receives from that participant, before any of our audio processing.
      const net = following.current ? await call.current?.audioStats(following.current) : null;
      if (net) {
        const rate = Math.max(0, net.bytes - lastBytes) * 2; // sampled every 0.5 s
        lastBytes = net.bytes;
        lines.push(
          `network: ${(rate / 1000).toFixed(1)} kB/s in · WebRTC level ${net.level == null ? "n/a" : (net.level * 1000).toFixed(1)}${net.paused ? " · PAUSED" : ""}`
        );
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
    tracks.current.clear();
    followedTrack.current = null;
    camera.current?.getTracks().forEach((t) => t.stop());
    camera.current = null;
    stopMeter.current?.();
    stopMeter.current = null;
  };

  const meter = (track: MediaStreamTrack) => {
    stopMeter.current?.();
    stopMeter.current = micMeter(track, (level) => {
      micLevelRef.current = level;
      setMicLevel(level);
    });
  };

  /** Switches the microphone live in the call, exactly to the chosen device. */
  const switchMic = async (deviceId: string) => {
    let mic;
    try {
      mic = await openMic(deviceId);
    } catch (err) {
      return setStatus(`Can't open that microphone: ${err instanceof Error ? err.message : err}`);
    }
    const [track] = mic.stream.getAudioTracks();
    const media = camera.current;
    media?.getAudioTracks().forEach((t) => {
      t.stop();
      media.removeTrack(t);
    });
    media?.addTrack(track);
    await call.current?.replaceMicTrack(track);
    saveMicId(mic.deviceId);
    setMicId(mic.deviceId);
    meter(track);
    console.log("[demo-bot] microphone:", mic.label);
  };

  // Render the background and idle face as soon as the page opens, so nothing changes later.
  useEffect(() => {
    const sync = new Avtr1LipSync(canvas.current!, AVATAR, setLive, setStatus, VOICE_PITCH);
    lipSync.current = sync;
    void sync.prepare().catch((err) => {
      if (err.name !== "AbortError") setStatus(String(err.message ?? err)); // aborted = page left
    });
    return () => {
      stopAll();
      sync.stop();
    };
  }, []);

  /** Plays a participant's voice as it is (no face). */
  const playPlain = (peerId: string, track: MediaStreamTrack) => {
    const player = new Audio();
    player.srcObject = new MediaStream([track]);
    void player.play().catch(() => {});
    players.current.set(peerId, player);
  };

  const stopPlain = (peerId: string) => {
    const player = players.current.get(peerId);
    if (player) player.srcObject = null;
    players.current.delete(peerId);
  };

  /**
   * Decides again whose voice drives Geeta: the assistant's (by name), else the only
   * participant's. Runs whenever tracks or names change, since the assistant's name can
   * arrive after its audio, and it can get a fresh audio track mid-call.
   */
  const pickVoice = () => {
    const isAssistant = (id: string) => ASSISTANT.test(peers.current.get(id) ?? "");
    const ids = [...tracks.current.keys()];
    const want =
      ids.find(isAssistant) ??
      (following.current && tracks.current.has(following.current) ? following.current : undefined) ??
      (ids.length === 1 && peers.current.size <= 1 ? ids[0] : undefined);
    const track = want ? tracks.current.get(want) : undefined;
    if (want === following.current && track === followedTrack.current) return;
    // The previous voice (if still there) goes back to a plain player.
    const prev = following.current;
    if (prev && prev !== want && tracks.current.has(prev)) playPlain(prev, tracks.current.get(prev)!);
    lipSync.current?.release();
    following.current = want ?? null;
    followedTrack.current = track ?? null;
    if (!want || !track) return;
    stopPlain(want);
    console.log("[demo-bot] Geeta follows", peers.current.get(want) || want);
    void lipSync.current?.start(track);
  };

  const onRemoteTrack = (peerId: string, kind: "audio" | "video", track: MediaStreamTrack | null) => {
    if (kind !== "audio") return; // the assistant has no video; others' isn't shown here
    if (track) audioPeers.current.add(peerId);
    else audioPeers.current.delete(peerId);
    stopPlain(peerId);
    if (track) {
      console.log("[demo-bot] audio from", peers.current.get(peerId) || peerId);
      tracks.current.set(peerId, track);
      playPlain(peerId, track); // until pickVoice takes it over
    } else {
      tracks.current.delete(peerId);
    }
    pickVoice();
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
    const video = await navigator.mediaDevices.getUserMedia({ video: true }).catch(() => null);
    const mic = await chooseMic().catch((err) => {
      setStatus(`No microphone: ${err instanceof Error ? err.message : err}. The assistant won't hear you.`);
      return null;
    });
    const media = new MediaStream([...(mic?.stream.getAudioTracks() ?? []), ...(video?.getVideoTracks() ?? [])]);
    camera.current = media;
    if (self.current) self.current.srcObject = media;
    if (mic) {
      console.log("[demo-bot] microphone:", mic.label);
      if (mic.note) setStatus(mic.note);
      setMicId(mic.deviceId);
      setMics(await microphones());
      meter(mic.stream.getAudioTracks()[0]);
    }

    const c = new VConfCall(
      SERVER,
      jwt,
      null,
      {
        onStatus: setStatus,
        onPeers: (p) => {
          peers.current = p;
          console.log("[demo-bot] in the call with", [...p.values()]);
          pickVoice(); // e.g. the assistant's name arrived after its audio
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
      {started && !audio && mics.length > 0 && (
        <div className="demo-bot-mic">
          <select value={micId} onChange={(e) => void switchMic(e.target.value)} aria-label="Microphone">
            {mics.map((m, i) => (
              <option key={m.deviceId} value={m.deviceId}>
                {m.label || `Microphone ${i + 1}`}
              </option>
            ))}
          </select>
          {/* Speech is roughly 0.02-0.2 RMS; the bar fills at 0.15. */}
          <div className="demo-bot-meter" title="Your microphone level">
            <div style={{ width: `${Math.min(100, (micLevel / 0.15) * 100)}%` }} />
          </div>
        </div>
      )}
      {(jwt || audio) && !started && (
        <button className="demo-bot-start" onClick={start}>
          Start
        </button>
      )}
      {debug && <pre className="demo-bot-debug">{debugText}</pre>}
      {(!jwt && !audio) || status ? (
        <div className="demo-bot-status">{status ?? "Open an instant-meeting link (/demo-bot/instant-<id>?t=<token>) to join a call."}</div>
      ) : null}
    </div>
  );
}
