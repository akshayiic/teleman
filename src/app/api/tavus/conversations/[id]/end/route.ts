import { errorResponse, tavus } from "@/server/tavus";

/** Ends a Tavus conversation so it stops billing. */
export async function POST(_request: Request, ctx: RouteContext<"/api/tavus/conversations/[id]/end">) {
  const { id } = await ctx.params;
  try {
    await tavus(`/v2/conversations/${encodeURIComponent(id)}/end`, { method: "POST" });
    return Response.json({ ok: true });
  } catch (err) {
    return errorResponse(err);
  }
}
