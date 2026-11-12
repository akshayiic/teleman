"use client";

import { useEffect, useImperativeHandle, useRef, type Ref } from "react";

/**
 * Realistic avatar with real lip-sync:
 * - TalkingHead (github.com/met4citizen/TalkingHead) renders a GLB avatar that
 *   has ARKit + Oculus viseme blend shapes, with idle motion, eye contact,
 *   blinking and gestures.
 * - HeadTTS (github.com/met4citizen/HeadTTS) runs the Kokoro neural TTS model
 *   in the browser (WebGPU, or WASM as a fallback) and returns the audio plus
 *   phoneme-level viseme timings, which TalkingHead plays back in sync.
 *
 * HeadTTS is served from /public/headtts (its npm package is Node-oriented),
 * so it is loaded at runtime rather than bundled.
 */

export type AvatarConfig = {
  url: string;
  body: "M" | "F";
  mood?: string;
  baseline?: Record<string, number>;
  retarget?: Record<string, unknown>;
  /** Base colour multiplied into the texture of these materials (by name), e.g. to change skin tone. */
  tint?: Record<string, string>;
};

export type TalkingAvatarHandle = {
  speak(text: string): void;
  stop(): void;
  /** What the presenter shows and says, as a live stream (e.g. to send into a video call). */
  outputStream?(): MediaStream;
};

type TalkingAvatarProps = {
  ref?: Ref<TalkingAvatarHandle>;
  avatar: AvatarConfig;
  voice: string;
  /** Progress/error text per subsystem; null clears it. */
  onStatus(source: "avatar" | "voice", status: string | null): void;
  onSpeakingChange(speaking: boolean): void;
};

/* eslint-disable @typescript-eslint/no-explicit-any -- both libraries are untyped JS */
type Head = any;
type Tts = any;

export default function TalkingAvatar({ ref, avatar, voice, onStatus, onSpeakingChange }: TalkingAvatarProps) {
  const container = useRef<HTMLDivElement>(null);
  const head = useRef<Head>(null);
  const tts = useRef<Tts>(null);
  // Latest callbacks, so the long-lived library handlers never go stale.
  const callbacks = useRef({ onStatus, onSpeakingChange });
  useEffect(() => {
    callbacks.current = { onStatus, onSpeakingChange };
  });

  // Create the TalkingHead instance and load the avatar.
  useEffect(() => {
    let cancelled = false;
    let instance: Head = null;

    (async () => {
      callbacks.current.onStatus("avatar", "Loading avatar…");
      const [{ TalkingHead }, { LipsyncEn }] = await Promise.all([
        import("@met4citizen/talkinghead"),
        import("@met4citizen/talkinghead/modules/lipsync-en.mjs"),
      ]);
      if (cancelled || !container.current) return;

      instance = new TalkingHead(container.current, {
        ttsEndpoint: null, // speech comes from HeadTTS, not Google TTS
        lipsyncModules: [], // its dynamic import() of these doesn't survive bundling…
        lipsyncLang: "en",
        cameraView: "upper",
        cameraRotateEnable: true,
        avatarMood: avatar.mood ?? "neutral",
      });
      instance.lipsync.en = new LipsyncEn(); // …so register English lip-sync directly.

      await instance.showAvatar(
        {
          url: avatar.url,
          body: avatar.body,
          avatarMood: avatar.mood ?? "neutral",
          lipsyncLang: "en",
          baseline: avatar.baseline,
          retarget: avatar.retarget,
        },
        (ev: ProgressEvent) => {
          if (ev.lengthComputable) {
            callbacks.current.onStatus("avatar", `Loading avatar… ${Math.round((ev.loaded / ev.total) * 100)}%`);
          }
        }
      );
      if (cancelled) {
        instance.dispose?.();
        return;
      }
      if (avatar.tint) {
        instance.armature.traverse((o: any) => {
          const color = avatar.tint?.[o.material?.name];
          if (color) o.material.color.set(color);
        });
      }
      head.current = instance;
      callbacks.current.onStatus("avatar", null);
    })().catch((err) => {
      if (!cancelled) callbacks.current.onStatus("avatar", `Avatar failed to load: ${err?.message ?? err}`);
    });

    return () => {
      cancelled = true;
      head.current = null;
      instance?.stop?.();
      instance?.dispose?.();
    };
  }, [avatar]);

  // Connect HeadTTS once; the voice is switched with setup().
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
        voices: [voice],
        defaultVoice: voice,
        dtypeWasm: "q8",
      });
      client.onmessage = (message: { type: string; data: any }) => {
        if (message.type === "audio") {
          head.current?.speakAudio(message.data, {}, null);
        } else if (message.type === "error") {
          callbacks.current.onStatus("voice", `Speech failed: ${message.data?.error ?? "unknown error"}`);
        }
      };
      client.onstart = () => callbacks.current.onSpeakingChange(true);
      client.onend = () => callbacks.current.onSpeakingChange(false);

      callbacks.current.onStatus("voice", "Loading voice model (first time only, ~100–300 MB)…");
      await client.connect(null, (ev: ProgressEvent) => {
        if (ev.lengthComputable && !cancelled) {
          callbacks.current.onStatus("voice", `Loading voice model… ${Math.round((ev.loaded / ev.total) * 100)}%`);
        }
      });
      if (cancelled) return;
      tts.current = client;
      callbacks.current.onStatus("voice", null);
    })().catch((err) => {
      if (!cancelled) callbacks.current.onStatus("voice", `Voice failed to load: ${err?.message ?? err}`);
    });
    return () => {
      cancelled = true;
      tts.current = null;
      // HeadTTS has no close(); stop its worker so a remount doesn't load the model twice.
      client?.ww?.terminate();
    };
    // The voice is applied in the effect below; reconnecting isn't needed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    tts.current?.setup({ voice, language: "en-us", speed: 1, audioEncoding: "wav" });
  }, [voice]);

  useImperativeHandle(ref, () => ({
    speak(text: string) {
      if (!tts.current) {
        callbacks.current.onStatus("voice", "The voice model is still loading, try again in a moment…");
        return;
      }
      tts.current.setup({ voice, language: "en-us", speed: 1, audioEncoding: "wav" });
      tts.current.synthesize({ input: text });
    },
    stop() {
      head.current?.stopSpeaking();
      callbacks.current.onSpeakingChange(false);
    },
  }));

  return <div ref={container} className="talking-avatar" />;
}
