"use client";

import dynamic from "next/dynamic";

// The WebGL scene and speech APIs only exist in the browser.
const Stage = dynamic(() => import("./Stage"), {
  ssr: false,
  loading: () => <div className="loading">Loading scene…</div>,
});

export default function StageLoader() {
  return <Stage />;
}
