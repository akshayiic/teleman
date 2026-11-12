import type { Metadata } from "next";
import InstantMeetingLoader from "@/components/InstantMeetingLoader";

export const metadata: Metadata = {
  title: "Geeta – instant meeting",
};

/**
 * Instant-meeting links, /demo-bot/instant-<meeting id>?t=<customer token>: the same
 * call as /demo-bot. Everything, including the path check, runs in the browser, so the
 * page is a static shell with nothing URL-dependent to prerender.
 */
export default function InstantMeetingPage() {
  return (
    <main>
      <InstantMeetingLoader />
    </main>
  );
}
