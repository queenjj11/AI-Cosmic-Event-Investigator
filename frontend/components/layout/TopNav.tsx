"use client";

import React, { useEffect, useState } from "react";
import { Activity, ShieldCheck, Database, Clock } from "lucide-react";

export function TopNav() {
  const [utcTime, setUtcTime] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().replace("GMT", "UTC"));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-14 border-b border-zinc-800 bg-[#0a0e17]/95 backdrop-blur-md px-4 flex items-center justify-between z-30 sticky top-0">
      {/* Brand & Wordmark */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-gradient-to-br from-cyan-600 to-blue-800 flex items-center justify-center font-mono font-bold text-white shadow-lg shadow-cyan-950/40 border border-cyan-500/40">
          ✦
        </div>
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <span className="font-mono font-black tracking-wider text-base text-white">ACEI</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
              v1.0 LC-First
            </span>
          </div>
          <span className="text-[10px] tracking-widest uppercase text-zinc-400 font-medium">
            AI Cosmic Event Investigator
          </span>
        </div>
      </div>

      {/* Observational Instrumentation Telemetry */}
      <div className="hidden md:flex items-center gap-4 text-xs font-mono text-zinc-400">
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-900/80 border border-zinc-800">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>CHECKPOINT: <span className="text-emerald-400 font-semibold">FROZEN (e5c7...bc72)</span></span>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-900/80 border border-zinc-800">
          <Database className="w-3.5 h-3.5 text-cyan-400" />
          <span>SURVEY: <span className="text-zinc-200">ZTF / NASA IRSA</span></span>
        </div>

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-900/80 border border-zinc-800">
          <Activity className="w-3.5 h-3.5 text-blue-400" />
          <span>ENCODER: <span className="text-blue-400 font-semibold">128-D TRANSFORMER</span></span>
        </div>
      </div>

      {/* Live UTC Clock */}
      <div className="flex items-center gap-2 font-mono text-xs text-zinc-300 bg-zinc-900/90 px-3 py-1.5 rounded border border-zinc-800">
        <Clock className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
        <span>{utcTime || "UTC CLOCK"}</span>
      </div>
    </header>
  );
}
