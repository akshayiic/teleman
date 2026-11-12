import type { Metadata } from "next";
import DemoBotLoader from "@/components/DemoBotLoader";

export const metadata: Metadata = {
  title: "Geeta – demo bot",
};

export default function DemoBotPage() {
  return (
    <main>
      <DemoBotLoader />
    </main>
  );
}
