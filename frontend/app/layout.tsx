import type { Metadata } from "next";
import dynamic from "next/dynamic";
import "./globals.css";
import { TopNav } from "@/components/layout/TopNav";
import { Sidebar } from "@/components/layout/Sidebar";

const ScientificField = dynamic(() => import("@/components/three/ScientificField"), {
  ssr: false,
  loading: () => <div className="fixed inset-0 pointer-events-none z-0 bg-[#07090e] scientific-grid opacity-30" />,
});

export const metadata: Metadata = {
  title: "ACEI — AI Cosmic Event Investigator",
  description: "Mission control and observatory workstation for AI-powered astronomical transient investigation on real ZTF observations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#07090e] text-slate-100 min-h-screen flex flex-col font-sans antialiased scientific-grid selection:bg-cyan-500/20 selection:text-cyan-200">
        <ScientificField />
        <TopNav />
        <div className="flex flex-1 overflow-hidden z-10 relative">
          <Sidebar />
          <main className="flex-1 overflow-y-auto min-h-[calc(100vh-3.5rem)] bg-[#07090e]/90">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
