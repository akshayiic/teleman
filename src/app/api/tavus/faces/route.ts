import { errorResponse, tavus, type TavusFace } from "@/server/tavus";

/** Preview videos for the requested Tavus faces: GET /api/tavus/faces?ids=a,b */
export async function GET(request: Request) {
  const ids = new URL(request.url).searchParams.get("ids") ?? "";
  if (!/^[\w,]+$/.test(ids)) {
    return Response.json({ error: "Pass face IDs as ?ids=a,b" }, { status: 400 });
  }
  try {
    const { data } = await tavus<{ data: TavusFace[] }>(`/v2/faces?face_ids=${ids}`);
    return Response.json(
      Object.fromEntries(data.map((f) => [f.face_id, { name: f.face_name, preview: f.thumbnail_video_url ?? null }]))
    );
  } catch (err) {
    return errorResponse(err);
  }
}
