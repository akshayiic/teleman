"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { VConfCall } from "@/lib/vconf";
import type { TalkingAvatarHandle } from "./TalkingAvatar";

/**
 * Puts the presenter into a V-Retail conference call as the agent: her video and
 * voice go into the room, and she says whatever text the backend sends over the
 * socket. Other backend messages are listed, to see what it sends.
 */

const SERVER = process.env.NEXT_PUBLIC_VCONF_SERVER ?? "https://backend-1.conference.svc.inkincaps.com";

type LogLine = { from: string; text: string };

type CallPanelProps = {
  presenter: RefObject<TalkingAvatarHandle | null>;
  /** Display name in the call. */
  name: string;
};

/** The words to say, when a backend message carries them. */
function spokenText(data: unknown) {
  if (typeof data === "string") return data;
  if (data && typeof data === "object") {
    const { text, message } = data as { text?: unknown; message?: unknown };
    if (typeof text === "string") return text;
    if (typeof message === "string") return message;
  }
  return null;
}

export default function CallPanel({ presenter, name }: CallPanelProps) {
  const [jwt, setJwt] = useState("");
  const [state, setState] = useState<"idle" | "joining" | "joined">("idle");
  const [status, setStatus] = useState<string | null>(null);
  const [peers, setPeers] = useState<Map<string, string>>(new Map());
  const [log, setLog] = useState<LogLine[]>([]);
  const call = useRef<VConfCall | null>(null);
  const logBox = useRef<HTMLDivElement>(null);

  const leave = () => {
    call.current?.close();
    call.current = null;
  };

  useEffect(() => logBox.current?.scrollTo(0, logBox.current.scrollHeight), [log]);
  useEffect(() => leave, []);

  const onNotification = (method: string, data: unknown) => {
    const text = spokenText(data)?.trim();
    if (text) {
      setLog((l) => [...l, { from: `${name} (${method})`, text }]);
      presenter.current?.speak(text);
    } else {
      setLog((l) => [...l, { from: method, text: JSON.stringify(data ?? null).slice(0, 300) }]);
    }
  };

  const join = async () => {
    const stream = presenter.current?.outputStream?.();
    if (!stream) return setStatus("Choose a local presenter first.");
    setState("joining");
    setLog([]);
    const c = new VConfCall(SERVER, jwt.trim(), name, {
      onStatus: setStatus,
      onPeers: setPeers,
      onNotification,
      onClosed: (reason) => {
        if (call.current === c) call.current = null;
        setState("idle");
        setPeers(new Map());
        setStatus(reason === "Left the call." ? null : reason);
      },
    });
    call.current = c;
    try {
      await c.join(stream);
      setState("joined");
    } catch {
      // onClosed already reported why.
    }
  };

  return (
    <div className="call">
      <div className="row">
        <input
          type="password"
          value={jwt}
          onChange={(e) => setJwt(e.target.value)}
          placeholder="Agent JWT for the conference"
          disabled={state !== "idle"}
          aria-label="Agent JWT"
        />
        {state === "idle" ? (
          <button onClick={join} disabled={!jwt.trim()}>
            Join call
          </button>
        ) : (
          <button onClick={leave}>{state === "joining" ? "Cancel" : "Leave call"}</button>
        )}
      </div>
      {state === "joined" && (
        <div className="call-peers">
          {name} is in the call with {peers.size ? [...peers.values()].join(", ") : "nobody else yet"}
        </div>
      )}
      {status && <div className="status">{status}</div>}
      {log.length > 0 && (
        <div className="call-log" ref={logBox}>
          {log.map((line, i) => (
            <div key={i}>
              <b>{line.from}:</b> {line.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
