"use client";

import { useCallback, useRef, useState } from "react";
import TalkingAvatar, { type AvatarConfig, type TalkingAvatarHandle } from "./TalkingAvatar";
import TavusLive from "./TavusLive";
import LiteAvatarPresenter from "./LiteAvatarPresenter";
import CallPanel from "./CallPanel";

/**
 * Real-time real-human presenters from Tavus (live stream): the best stock woman and man
 * (Phoenix-4 "Professional"/"Office" faces, picked for natural look).
 * `idleAt` is a moment in the face's preview clip where the mouth is closed,
 * shown as a still while nothing is being read.
 */
const TAVUS = {
  anna: { label: "Anna (real, live)", faceId: "rf4e9d9790f0", idleAt: 8.5 }, // Anna - Professional
  daniel: { label: "Daniel (real, live)", faceId: "r72f7f7f7c8b", idleAt: 0 }, // Daniel - Office
};

/** Pose fixes for the MetaPerson skeleton (Avatar SDK), shared by all MetaPerson-based avatars. */
const METAPERSON_RETARGET = {
  Neck: { z: -0.01, rx: -0.15 },
  Neck1: { z: -0.01, rx: -0.15 },
  Neck2: { z: -0.01, rx: -0.15 },
  LeftShoulder: { rz: -0.3 },
  RightShoulder: { rz: 0.3 },
  scaleToEyesLevel: 1.0,
  origin: { y: -0.1 },
};

/**
 * Realistic avatars with ARKit/viseme blend shapes, voiced by HeadTTS (Kokoro).
 * Kavita and Arjun are Sketchfab MetaPersons whose blend shapes were restored with
 * scripts/patch-metaperson-avatar.mjs (copied from Michael).
 */
const AVATARS = {
  michael: {
    label: "Michael (3D, instant)",
    avatar: {
      url: "/avatars/avatarsdk.glb",
      body: "M",
      mood: "neutral",
      retarget: METAPERSON_RETARGET,
      baseline: { headRotateX: -0.04, eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 },
      tint: { AvatarHead: "#b8906f", AvatarBody: "#b8906f" }, // light-brown skin
    },
    voice: "am_michael",
    credit: "Avatar created with Avatar SDK (MetaPerson), non-commercial use",
  },
  arjun: {
    label: "Arjun (3D, instant)",
    avatar: {
      url: "/avatars/arjun.glb",
      body: "M",
      mood: "neutral",
      retarget: METAPERSON_RETARGET,
      baseline: { headRotateX: -0.04, eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 },
    },
    voice: "hm_omega",
    credit: '"Indian Man in suit" by Pixel_Monster (Sketchfab, CC BY 4.0); face shapes from Avatar SDK, non-commercial',
  },
  kavita: {
    label: "Kavita (3D, instant)",
    avatar: {
      url: "/avatars/kavita.glb",
      body: "F",
      mood: "happy",
      retarget: METAPERSON_RETARGET,
      baseline: { headRotateX: -0.04, eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 },
    },
    voice: "hf_alpha",
    credit: '"Indian Woman in Saree" by Pixel_Monster (Sketchfab); face shapes from Avatar SDK, non-commercial',
  },
} satisfies Record<string, { label: string; avatar: AvatarConfig; voice: string; credit: string }>;

/** Kokoro voices (huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX#voices). */
const VOICES = {
  M: {
    hm_omega: "Omega (Indian)",
    hm_psi: "Psi (Indian)",
    am_michael: "Michael (US)",
    am_fenrir: "Fenrir (US)",
    am_puck: "Puck (US)",
    bm_george: "George (UK)",
  },
  F: {
    hf_alpha: "Alpha (Indian)",
    hf_beta: "Beta (Indian)",
    af_heart: "Heart (US)",
    af_bella: "Bella (US)",
    af_nicole: "Nicole (US)",
    bf_emma: "Emma (UK)",
  },
};

/** Free, unlimited photoreal presenters rendered locally (LiteAvatar + Piper voice). */
const LOCAL_CREDIT = "LiteAvatar (MIT) + Piper voice, rendered on this PC";
const LOCAL = {
  priti: { label: "Priti (your photo, free, local)", credit: "LivePortrait (MIT) + Piper voice, rendered on this PC" },
  "priti-hd": { label: "Priti HD (AVTR-1, GPU server)", credit: "AVTR-1 (non-commercial) on a GPU server + Piper voice" },
  "aditi-hd": { label: "Aditi HD (AVTR-1, GPU server)", credit: "AVTR-1 (non-commercial) on a GPU server + Piper voice" },
  "sahil-hd": { label: "Sahil HD (AVTR-1, GPU server)", credit: "AVTR-1 (non-commercial) on a GPU server + Piper voice" },
  priya: { label: "Priya (real, free, local)", credit: LOCAL_CREDIT },
  kavya: { label: "Kavya (real, free, local)", credit: LOCAL_CREDIT },
};

/** Piper voices on the local server; "auto" reads Devanagari text in Hindi. */
const LOCAL_LANGS = { auto: "Auto language", en: "English", hi: "हिन्दी (Hindi)" };
type LocalLang = keyof typeof LOCAL_LANGS;

type TavusKey = keyof typeof TAVUS;
type AvatarKey = keyof typeof AVATARS;
type LocalKey = keyof typeof LOCAL;
type Key = TavusKey | AvatarKey | LocalKey;
const isTavus = (key: Key): key is TavusKey => key in TAVUS;
const isLocal = (key: Key): key is LocalKey => key in LOCAL;

export default function Stage() {
  const [text, setText] = useState(
    "Hello! I am a realistic digital human. Type anything you like, and I will read it out loud for you."
  );
  const [key, setKey] = useState<Key>("anna");
  const [speaking, setSpeaking] = useState(false);
  const [statuses, setStatuses] = useState<Record<string, string | null>>({});
  const status = Object.values(statuses).filter(Boolean).join(" · ") || null;
  const setStatus = useCallback(
    (source: string, text: string | null) => setStatuses((s) => ({ ...s, [source]: text })),
    []
  );

  const tavusEntry = isTavus(key) ? TAVUS[key] : null;
  const localEntry = isLocal(key) ? LOCAL[key] : null;
  const avatarEntry = isTavus(key) || isLocal(key) ? null : AVATARS[key];
  const body = avatarEntry?.avatar.body ?? "F";
  const [voice, setVoice] = useState("am_michael");
  const [localLang, setLocalLang] = useState<LocalLang>("auto");
  const presenter = useRef<TalkingAvatarHandle>(null);

  const speak = () => {
    if (text.trim()) presenter.current?.speak(text);
  };

  const stop = () => {
    presenter.current?.stop();
    setSpeaking(false);
  };

  const select = (next: Key) => {
    stop();
    setStatuses({});
    setKey(next);
    if (!isTavus(next) && !isLocal(next)) setVoice(AVATARS[next].voice);
  };

  return (
    <div className="stage">
      {avatarEntry ? (
        <TalkingAvatar
          key={key}
          ref={presenter}
          avatar={avatarEntry.avatar}
          voice={voice}
          onStatus={setStatus}
          onSpeakingChange={setSpeaking}
        />
      ) : localEntry ? (
        <LiteAvatarPresenter
          key={key}
          ref={presenter}
          avatar={key}
          lang={localLang}
          onStatus={setStatus}
          onSpeakingChange={setSpeaking}
        />
      ) : (
        tavusEntry && (
          <TavusLive
            key={key}
            ref={presenter}
            faceId={tavusEntry.faceId}
            idleAt={tavusEntry.idleAt}
            onStatus={setStatus}
            onSpeakingChange={setSpeaking}
          />
        )
      )}

      <div className="credit">{avatarEntry?.credit ?? localEntry?.credit ?? "Live video by Tavus"}</div>

      <div className="panel">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          placeholder="Type something for them to say…"
        />
        <div className="row">
          {speaking ? <button onClick={stop}>Stop</button> : <button onClick={speak}>Speak</button>}
          <select value={key} onChange={(e) => select(e.target.value as Key)}>
            {Object.entries({ ...TAVUS, ...LOCAL, ...AVATARS }).map(([k, a]) => (
              <option key={k} value={k}>
                {a.label}
              </option>
            ))}
          </select>
          {avatarEntry && (
            <select value={voice} onChange={(e) => setVoice(e.target.value)}>
              {Object.entries(VOICES[body]).map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </select>
          )}
          {localEntry && (
            <select value={localLang} onChange={(e) => setLocalLang(e.target.value as LocalLang)}>
              {Object.entries(LOCAL_LANGS).map(([id, name]) => (
                <option key={id} value={id}>
                  {name}
                </option>
              ))}
            </select>
          )}
        </div>
        {status && <div className="status">{status}</div>}
        {localEntry && <CallPanel key={key} presenter={presenter} name={localEntry.label.split(" ")[0]} />}
      </div>
    </div>
  );
}
