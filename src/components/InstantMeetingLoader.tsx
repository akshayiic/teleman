"use client";

import dynamic from "next/dynamic";

// Media, Web Audio, the call client and the URL check only exist in the browser.
const InstantMeeting = dynamic(() => import("./InstantMeeting"), {
  ssr: false,
  loading: () => <div className="loading">Loading…</div>,
});

export default function InstantMeetingLoader() {
  return <InstantMeeting />;
}
