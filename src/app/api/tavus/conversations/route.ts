import { errorResponse, tavus, TavusError } from "@/server/tavus";

/**
 * Starts a real-time Tavus conversation (CVI) for a face, using the account's
 * echo-mode PAL so the face speaks exactly the text we send it, live.
 * The browser joins `url` with Daily and sends `conversation.echo` messages.
 */
export async function POST(request: Request) {
  const { faceId } = await request.json().catch(() => ({}));
  if (typeof faceId !== "string" || !/^\w+$/.test(faceId)) {
    return Response.json({ error: "Missing faceId." }, { status: 400 });
  }
  const palId = process.env.TAVUS_ECHO_PAL_ID;

  try {
    if (!palId) throw new TavusError("Set TAVUS_ECHO_PAL_ID in .env.local (an echo-mode PAL).", 501);
    const conversation = await tavus<{ conversation_id: string; conversation_url: string }>("/v2/conversations", {
      method: "POST",
      body: JSON.stringify({
        face_id: faceId,
        pal_id: palId,
        conversation_name: `teleman ${new Date().toISOString()}`,
        // Billing is per minute, so keep sessions short and self-closing.
        properties: {
          max_call_duration: 600,
          participant_left_timeout: 5,
          participant_absent_timeout: 60,
        },
      }),
    });
    return Response.json({ id: conversation.conversation_id, url: conversation.conversation_url });
  } catch (err) {
    return errorResponse(err);
  }
}
