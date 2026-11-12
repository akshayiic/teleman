import { A2FError, MAX_SECONDS, SAMPLE_RATE, audioToFace } from "@/server/a2f/client";

/**
 * POST raw 16 kHz mono PCM16 audio (application/octet-stream) and get back the
 * Audio2Face-3D blendshape track as JSON.
 */
export async function POST(request: Request) {
  const pcm = Buffer.from(await request.arrayBuffer());
  if (pcm.length === 0) {
    return Response.json({ error: "No audio received." }, { status: 400 });
  }
  if (pcm.length > SAMPLE_RATE * 2 * MAX_SECONDS) {
    return Response.json({ error: `Audio must be ${MAX_SECONDS} seconds or shorter.` }, { status: 413 });
  }

  try {
    return Response.json(await audioToFace(pcm));
  } catch (err) {
    const status = err instanceof A2FError ? err.status : 500;
    const message = err instanceof Error ? err.message : "Audio2Face request failed.";
    return Response.json({ error: message }, { status });
  }
}
