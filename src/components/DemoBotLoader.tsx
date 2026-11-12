"use client";

import dynamic from "next/dynamic";

// Media, Web Audio and the call client only exist in the browser.
const DemoBot = dynamic(() => import("./DemoBot"), {
  ssr: false,
  loading: () => <div className="loading">Loading…</div>,
});

export default function DemoBotLoader() {
  return <DemoBot />;
}
