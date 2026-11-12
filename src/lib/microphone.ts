/**
 * Microphone choice for calls. A device id passed as a bare value is only a preference,
 * so the browser may hand back a different microphone than the one chosen; only
 * `{ exact }` makes it a requirement (the same fix as conference-front's 383399d, where
 * a call sent almost no sound and the AI bot in it never replied). Laptops often expose
 * both a jack input that records silence and a built-in "Microphone Array".
 */

const KEY = "demo-bot-mic";

export type Mic = { stream: MediaStream; deviceId: string; label: string };

export function savedMicId() {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function saveMicId(deviceId: string) {
  try {
    localStorage.setItem(KEY, deviceId);
  } catch {}
}

/** Opens a microphone; with an id, exactly that one (fails if it's gone rather than picking another). */
export async function openMic(deviceId?: string): Promise<Mic> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) },
  });
  const [track] = stream.getAudioTracks();
  return { stream, deviceId: track.getSettings().deviceId ?? deviceId ?? "", label: track.label };
}

/** Real microphones (labels are only filled in once the page has microphone permission). */
export async function microphones() {
  return (await navigator.mediaDevices.enumerateDevices()).filter(
    (d) => d.kind === "audioinput" && d.deviceId !== "default" && d.deviceId !== "communications"
  );
}

/**
 * The microphone to use: the one chosen last time, else the built-in microphone array
 * over other inputs (a jack input often records silence), else the browser's default.
 * `note` says when the saved choice wasn't available.
 */
export async function chooseMic(): Promise<Mic & { note?: string }> {
  const saved = savedMicId();
  let note: string | undefined;
  if (saved) {
    try {
      return await openMic(saved);
    } catch {
      note = "The microphone chosen last time isn't available; using another one.";
    }
  }
  const mic = await openMic(); // also grants permission, so device labels can be read
  const array = (await microphones()).find((d) => /array/i.test(d.label));
  if (array && array.deviceId !== mic.deviceId) {
    mic.stream.getTracks().forEach((t) => t.stop());
    return { ...(await openMic(array.deviceId)), note };
  }
  return { ...mic, note };
}

/** Calls `onLevel` about 10 times a second with the track's input loudness (RMS, 0..1). Returns a stop function. */
export function micMeter(track: MediaStreamTrack, onLevel: (level: number) => void) {
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  ctx.createMediaStreamSource(new MediaStream([track])).connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  const timer = setInterval(() => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (const s of buf) sum += s * s;
    onLevel(Math.sqrt(sum / buf.length));
  }, 100);
  return () => {
    clearInterval(timer);
    void ctx.close();
  };
}
