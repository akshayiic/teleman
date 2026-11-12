"use client";

import dynamic from "next/dynamic";

// WebGL, Web Audio and the call client only exist in the browser.
const DemoBot3D = dynamic(() => import("./DemoBot3D"), {
  ssr: false,
  loading: () => <div className="loading">Loading…</div>,
});

export default function DemoBot3DLoader() {
  return <DemoBot3D />;
}
