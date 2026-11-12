/**
 * Minimal server-side client for the Tavus API (docs.tavus.io). Keeps the API
 * key on the server; the browser only talks to our /api/tavus routes.
 *
 * Real-time conversations: POST /v2/conversations with a face and an echo-mode
 * PAL, then the browser joins the returned Daily room.
 */

const BASE_URL = "https://tavusapi.com";

export class TavusError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
  }
}

export async function tavus<T>(path: string, init: RequestInit = {}): Promise<T> {
  const apiKey = process.env.TAVUS_API_KEY;
  if (!apiKey) {
    throw new TavusError("Set TAVUS_API_KEY in .env.local (from maker.tavus.io/dev/api-keys) and restart the dev server.", 501);
  }
  const res = await fetch(BASE_URL + path, {
    ...init,
    headers: { "Content-Type": "application/json", "x-api-key": apiKey, ...init.headers },
    cache: "no-store",
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new TavusError(`Tavus: ${body.error ?? body.message ?? res.statusText}`, res.status);
  }
  return body as T;
}

/** Converts thrown errors into a JSON response for route handlers. */
export function errorResponse(err: unknown) {
  const status = err instanceof TavusError ? err.status : 500;
  const message = err instanceof Error ? err.message : "Tavus request failed.";
  return Response.json({ error: message }, { status });
}

export type TavusFace = {
  face_id: string;
  face_name: string;
  thumbnail_video_url?: string;
  status: string;
};
