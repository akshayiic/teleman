"use client";

import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import Daily, { type DailyCall } from "@daily-co/daily-js";
import type { TalkingAvatarHandle } from "./TalkingAvatar";

/**
 * Real-time Tavus presenter (Conversational Video Interface, echo mode).
 * We start a live conversation whose PAL is in echo mode, join it with Daily
 * (WebRTC), show the face's live stream, and send `conversation.echo` events:
 * the face speaks the text within about a second, and stays quiet otherwise.
 *
 * Conversations are billed per minute (idle time included), so the call only
 * connects when Speak is clicked and disconnects IDLE_DISCONNECT_MS after the
 * face stops speaking; the next Speak reconnects.
 */

type TavusLiveProps = {
  ref?: Ref<TalkingAvatarHandle>;
  /** Tavus face (replica) ID. */
  faceId: string;
  /** Time (s) in the face's preview clip with the mouth closed, shown while not connected. */
  idleAt: number;
  onStatus(source: "tavus", status: string | null): void;
  onSpeakingChange(speaking: boolean): void;
};

type Phase = "idle" | "connecting" | "live";

const IDLE_DISCONNECT_MS = 30 * 1000;

// Daily allows one call object at a time (and Tavus limits concurrent
// conversations), so sessions are closed one after another and a new one
// starts only once the previous teardown has finished.
let teardown: Promise<void> = Promise.resolve();
const queueTeardown = (step: () => Promise<void>) => (teardown = teardown.then(step));
const afterTeardown = () => teardown;

export default function TavusLive({ ref, faceId, idleAt, onStatus, onSpeakingChange }: TavusLiveProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const videoEl = useRef<HTMLVideoElement>(null);
  const audioEl = useRef<HTMLAudioElement>(null);
  const session = useRef<{ call: DailyCall; conversationId: string; ready: boolean } | null>(null);
  const pending = useRef<string | null>(null); // text to speak once the face has joined
  const connecting = useRef<Promise<void> | null>(null);
  // Bumped by disconnect(); a connect() that finishes under an older value was
  // cancelled and must close what it opened (each open session is billed).
  const generation = useRef(0);
  const idleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const callbacks = useRef({ onStatus, onSpeakingChange });
  useEffect(() => {
    callbacks.current = { onStatus, onSpeakingChange };
  });

  // Closed-mouth still from the face's preview clip, shown until the live stream arrives.
  useEffect(() => {
    let cancelled = false;
    fetch(`/api/tavus/faces?ids=${faceId}`)
      .then(async (res) => {
        const body = await res.json();
        if (!res.ok) throw new Error(body.error ?? `Request failed (${res.status})`);
        if (!cancelled) setPreview(body[faceId]?.preview ?? null);
      })
      .catch((err) => !cancelled && callbacks.current.onStatus("tavus", err.message));
    return () => {
      cancelled = true;
    };
  }, [faceId]);

  const disconnect = () => {
    generation.current++;
    if (idleTimer.current) clearTimeout(idleTimer.current);
    pending.current = null;
    connecting.current = null;
    const s = session.current;
    session.current = null;
    setPhase("idle");
    callbacks.current.onSpeakingChange(false);
    if (!s) return;
    queueTeardown(async () => {
      await s.call.leave().catch(() => {});
      await s.call.destroy().catch(() => {});
      // End server-side too, so billing stops immediately.
      await endConversation(s.conversationId);
    });
  };

  const resetIdleTimer = () => {
    if (idleTimer.current) clearTimeout(idleTimer.current);
    idleTimer.current = setTimeout(disconnect, IDLE_DISCONNECT_MS);
  };

  const sendEcho = (text: string) => {
    const s = session.current;
    if (!s) return;
    s.call.sendAppMessage(
      {
        message_type: "conversation",
        event_type: "conversation.echo",
        conversation_id: s.conversationId,
        properties: { modality: "text", text, done: true },
      },
      "*"
    );
    resetIdleTimer();
  };

  const endConversation = (conversationId: string) =>
    fetch(`/api/tavus/conversations/${conversationId}/end`, { method: "POST", keepalive: true }).catch(() => {});

  const connect = () => {
    connecting.current ??= (async () => {
      const gen = generation.current;
      const stale = () => gen !== generation.current;
      setPhase("connecting");
      callbacks.current.onStatus("tavus", "Connecting live (about 10 seconds)…");
      try {
        // Tavus allows a limited number of concurrent conversations, so let any
        // previous session finish closing before starting a new one.
        await afterTeardown();
        if (stale()) return;
        const res = await fetch("/api/tavus/conversations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ faceId }),
        });
        const conv = await res.json();
        if (!res.ok) throw new Error(conv.error);
        if (stale()) {
          void endConversation(conv.id);
          return;
        }

        const call = Daily.createCallObject({ subscribeToTracksAutomatically: true });
        session.current = { call, conversationId: conv.id, ready: false };

        call.on("track-started", (ev) => {
          if (!ev?.participant || ev.participant.local) return;
          const stream = new MediaStream([ev.track]);
          if (ev.track.kind === "video" && videoEl.current) {
            videoEl.current.srcObject = stream;
          } else if (ev.track.kind === "audio" && audioEl.current) {
            audioEl.current.srcObject = stream;
            audioEl.current.play().catch(() => {});
          }
        });
        call.on("participant-joined", (ev) => {
          if (!ev || ev.participant.local || !session.current) return;
          session.current.ready = true;
          setPhase("live");
          callbacks.current.onStatus("tavus", null);
          resetIdleTimer();
          if (pending.current) {
            sendEcho(pending.current);
            pending.current = null;
          }
        });
        call.on("app-message", (ev) => {
          const type = (ev?.data as { event_type?: string } | undefined)?.event_type;
          if (type === "conversation.replica.started_speaking") callbacks.current.onSpeakingChange(true);
          if (type === "conversation.replica.stopped_speaking") {
            callbacks.current.onSpeakingChange(false);
            resetIdleTimer();
          }
        });
        call.on("left-meeting", () => session.current?.call === call && disconnect());
        call.on("error", (ev) => {
          callbacks.current.onStatus("tavus", `Live session error: ${ev?.errorMsg ?? "unknown error"}`);
          disconnect();
        });

        // We only watch and listen; no camera or microphone is needed.
        await call.join({ url: conv.url, startVideoOff: true, startAudioOff: true });
        callbacks.current.onStatus("tavus", session.current?.ready ? null : "Connected, waiting for the presenter to join…");
      } catch (err) {
        if (stale()) return;
        callbacks.current.onStatus("tavus", `Could not start the live session: ${err instanceof Error ? err.message : err}`);
        disconnect();
      }
    })();
    return connecting.current;
  };

  // Close the live session when this presenter is hidden or the face changes.
  useEffect(
    () => disconnect,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [faceId]
  );

  useImperativeHandle(ref, () => ({
    speak(text: string) {
      audioEl.current?.play().catch(() => {}); // within the click, so sound is allowed
      if (session.current?.ready) {
        sendEcho(text);
      } else {
        pending.current = text;
        void connect();
      }
    },
    stop() {
      const s = session.current;
      pending.current = null;
      s?.call.sendAppMessage(
        { message_type: "conversation", event_type: "conversation.interrupt", conversation_id: s.conversationId },
        "*"
      );
      callbacks.current.onSpeakingChange(false);
    },
  }));

  return (
    <div className="tavus">
      {preview && phase !== "live" && (
        <video
          key={preview}
          src={preview}
          muted
          playsInline
          preload="auto"
          onLoadedMetadata={(e) => {
            e.currentTarget.currentTime = idleAt;
          }}
        />
      )}
      <video ref={videoEl} autoPlay playsInline muted hidden={phase !== "live"} />
      <audio ref={audioEl} autoPlay />
      {phase === "live" && <div className="tavus-live">● Live</div>}
    </div>
  );
}
