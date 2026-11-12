"use client";

import DemoBot from "./DemoBot";

/**
 * /demo-bot/instant-<meeting id> opens the Geeta call (token in ?t=); other /demo-bot/...
 * paths are not pages. Browser-only (see InstantMeetingLoader), so it reads the path
 * straight from the location.
 */
export default function InstantMeeting() {
  if (!/^\/demo-bot\/instant-[\w-]+\/?$/.test(location.pathname)) {
    return <div className="loading">This page could not be found.</div>;
  }
  return <DemoBot />;
}
