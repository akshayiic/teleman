import type { Metadata } from "next";
import DemoBot3DLoader from "@/components/DemoBot3DLoader";

export const metadata: Metadata = {
  title: "3D bot (Avaturn)",
};

export default function ThreeDBotPage() {
  return (
    <main>
      <DemoBot3DLoader />
    </main>
  );
}
