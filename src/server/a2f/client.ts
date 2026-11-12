import fs from "node:fs";
import path from "node:path";
import tls from "node:tls";
import * as grpc from "@grpc/grpc-js";
import * as protoLoader from "@grpc/proto-loader";

/**
 * Minimal client for NVIDIA Audio2Face-3D (hosted on NVCF), using the protos
 * from github.com/NVIDIA/Audio2Face-3D-Samples (vendored in ./protos).
 *
 * Flow (bidirectional stream, A2FControllerService/ProcessAudioStream):
 *   -> AudioStreamHeader, AudioWithEmotion (PCM chunks)…, EndOfAudio
 *   <- AnimationDataStreamHeader (blendshape names), AnimationData…, Status
 */

const DEFAULT_URI = "grpc.nvcf.nvidia.com:443";
/**
 * Hosted model to use, from build.nvidia.com/nvidia/audio2face-3d:
 * Mark 8efc55f5-6f00-424e-afe9-26212cd2c630, Claire 0961a6da-fb9e-4f2e-8491-247e5fd7bf8d,
 * James 9327c39f-a361-4e02-bd72-e11b4c9b7b5e.
 */
const DEFAULT_FUNCTION_ID = "8efc55f5-6f00-424e-afe9-26212cd2c630";

/** 16 kHz mono 16-bit PCM is what the service expects. */
export const SAMPLE_RATE = 16000;
/** The hosted function rejects requests longer than about a minute. */
export const MAX_SECONDS = 60;

export type A2FTrack = {
  /** ARKit-style blendshape names, in the order of each frame's weights. */
  names: string[];
  /** Frames at ~30 fps; `t` is seconds from the start of the audio. */
  frames: { t: number; w: number[] }[];
};

type Stream = grpc.ClientDuplexStream<object, AnimationDataStream>;
type AnimationDataStream = {
  stream_part: "animation_data_stream_header" | "animation_data" | "event" | "status";
  animation_data_stream_header?: { skel_animation_header?: { blend_shapes: string[] } };
  animation_data?: {
    skel_animation?: { blend_shape_weights: { time_code: number; values: number[] }[] };
  };
  status?: { code: string | number; message: string };
};
type ServiceClient = grpc.Client & {
  ProcessAudioStream(metadata: grpc.Metadata): Stream;
};

let Service: (new (address: string, credentials: grpc.ChannelCredentials) => ServiceClient) | null =
  null;

function service() {
  if (Service) return Service;
  const dir = path.join(process.cwd(), "src/server/a2f/protos");
  const definition = protoLoader.loadSync(
    path.join(dir, "nvidia_ace.services.a2f_controller.v1.proto"),
    { includeDirs: [dir], keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true }
  );
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pkg = grpc.loadPackageDefinition(definition) as any;
  Service = pkg.nvidia_ace.services.a2f_controller.v1.A2FControllerService;
  return Service!;
}

/**
 * Trusted roots for the TLS connection: Node's bundled CAs, the OS store, and
 * optionally an extra PEM file (A2F_CA_CERT), e.g. a corporate HTTPS-inspection
 * CA such as Sophos, which grpc-js would otherwise reject.
 */
function rootCertificates() {
  // getCACertificates (Node 22.15+) is newer than the installed @types/node.
  const { getCACertificates } = tls as typeof tls & { getCACertificates?: (type: string) => string[] };
  const certs = [...tls.rootCertificates, ...(getCACertificates?.("system") ?? [])];
  const extra = process.env.A2F_CA_CERT;
  if (extra) certs.push(fs.readFileSync(extra, "utf8"));
  return Buffer.from(certs.join("\n"));
}

export class A2FError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
  }
}

/** Sends 16 kHz mono PCM16 audio to Audio2Face-3D and collects the blendshape track. */
export function audioToFace(pcm: Buffer): Promise<A2FTrack> {
  const apiKey = process.env.NVIDIA_API_KEY;
  const functionId = process.env.A2F_FUNCTION_ID || DEFAULT_FUNCTION_ID;
  if (!apiKey) {
    return Promise.reject(
      new A2FError("Set NVIDIA_API_KEY in .env.local (get one at build.nvidia.com) and restart the dev server.", 501)
    );
  }

  const Client = service();
  const client = new Client(process.env.A2F_URI ?? DEFAULT_URI, grpc.credentials.createSsl(rootCertificates()));
  const metadata = new grpc.Metadata();
  metadata.set("function-id", functionId);
  metadata.set("authorization", `Bearer ${apiKey}`);

  return new Promise<A2FTrack>((resolve, reject) => {
    const track: A2FTrack = { names: [], frames: [] };
    let failure: A2FError | null = null;
    const stream = client.ProcessAudioStream(metadata);

    stream.on("data", (msg: AnimationDataStream) => {
      if (msg.stream_part === "animation_data_stream_header") {
        track.names = msg.animation_data_stream_header?.skel_animation_header?.blend_shapes ?? [];
      } else if (msg.stream_part === "animation_data") {
        for (const f of msg.animation_data?.skel_animation?.blend_shape_weights ?? []) {
          track.frames.push({ t: f.time_code, w: f.values });
        }
      } else if (msg.stream_part === "status" && msg.status) {
        const { code, message } = msg.status;
        if (code === "ERROR" || code === 3) failure = new A2FError(`Audio2Face: ${message}`, 502);
      }
    });
    stream.on("error", (err: grpc.ServiceError) => {
      client.close();
      const status =
        err.code === grpc.status.UNAUTHENTICATED || err.code === grpc.status.PERMISSION_DENIED
          ? 401
          : err.code === grpc.status.RESOURCE_EXHAUSTED
            ? 429
            : err.code === grpc.status.INVALID_ARGUMENT
              ? 400
              : 502;
      reject(new A2FError(`Audio2Face: ${err.details || err.message}`, status));
    });
    stream.on("end", () => {
      client.close();
      if (failure) reject(failure);
      else resolve(track);
    });

    stream.write({
      audio_stream_header: {
        audio_header: {
          audio_format: "AUDIO_FORMAT_PCM",
          channel_count: 1,
          samples_per_second: SAMPLE_RATE,
          bits_per_sample: 16,
        },
      },
    });
    // Send one second of audio per message (the size is arbitrary).
    const chunk = SAMPLE_RATE * 2;
    for (let i = 0; i < pcm.length; i += chunk) {
      stream.write({ audio_with_emotion: { audio_buffer: pcm.subarray(i, i + chunk) } });
    }
    stream.write({ end_of_audio: {} });
    stream.end();
  });
}
