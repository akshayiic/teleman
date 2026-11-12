/**
 * Same-origin pass-through to the AVTR-1 bridge on the GPU machine
 * (liteavatar-server/avtr1_bridge.py at AVTR1_URL), for pages that drive an
 * AVTR-1 face from audio in the browser:
 *   POST /api/avtr1/render?avatar=&session=   int16 16 kHz PCM in, streamed frames out
 *   GET  /api/avtr1/idle.webm?avatar=         a few seconds of the face idling
 */

const ACTIONS = { render: "/render", "idle.webm": "/idle.webm" } as const;

async function forward(request: Request, action: string) {
  const path = ACTIONS[action as keyof typeof ACTIONS];
  const bridge = process.env.AVTR1_URL?.replace(/\/$/, "");
  if (!path) return Response.json({ error: "not found" }, { status: 404 });
  if (!bridge) return Response.json({ error: "Set AVTR1_URL in .env.local (the AVTR-1 bridge)." }, { status: 501 });
  const { search } = new URL(request.url);
  try {
    const res = await fetch(`${bridge}${path}${search}`, {
      method: request.method,
      headers: { "X-Tunnel-Skip-AntiPhishing-Page": "true" },
      body: request.method === "POST" ? await request.arrayBuffer() : undefined,
      // When the browser gives up on a render, drop it on the bridge too, or the bridge
      // keeps writing frames nobody reads while holding its one-render-at-a-time GPU lock.
      signal: request.signal,
    });
    return new Response(res.body, {
      status: res.status,
      headers: { "Content-Type": res.headers.get("Content-Type") ?? "application/octet-stream", "Cache-Control": "no-store" },
    });
  } catch (err) {
    return Response.json({ error: `Can't reach the AVTR-1 bridge at ${bridge} (${err instanceof Error ? err.message : err}).` }, { status: 502 });
  }
}

export async function GET(request: Request, ctx: RouteContext<"/api/avtr1/[action]">) {
  return forward(request, (await ctx.params).action);
}

export async function POST(request: Request, ctx: RouteContext<"/api/avtr1/[action]">) {
  return forward(request, (await ctx.params).action);
}
