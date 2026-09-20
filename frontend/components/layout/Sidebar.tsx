"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Compass,
  Search,
  BookOpen,
  CalendarClock,
  FileText,
  Server,
  ExternalLink
} from "lucide-react";

export function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { label: "Overview", href: "/dashboard", icon: LayoutDashboard },
    { label: "Investigate", href: "/investigate/CAND_SNIa_002", icon: Compass },
    { label: "Candidate Browser", href: "/candidates", icon: Search },
    { label: "Knowledge Base", href: "/knowledge", icon: BookOpen },
    { label: "Observation Planner", href: "/planner", icon: CalendarClock },
    { label: "Investigation Reports", href: "/reports", icon: FileText },
    { label: "System Architecture", href: "/system", icon: Server },
  ];

  return (
    <aside className="w-60 border-r border-zinc-800 bg-[#090d16] flex flex-col justify-between shrink-0 min-h-[calc(100vh-3.5rem)]">
      {/* Navigation Links */}
      <div className="p-3 space-y-1">
        <div className="px-3 py-2 text-[10px] font-mono tracking-wider text-zinc-500 uppercase">
          Observatory Console
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href.split("/")[1] ? `/${item.href.split("/")[1]}` : "none"));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? "bg-cyan-950/50 text-cyan-300 border border-cyan-800/50"
                  : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50"
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : "text-zinc-400"}`} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </div>

      {/* Internal Research Mode & Telemetry */}
      <div className="p-3 border-t border-zinc-800/80 space-y-2">
        <a
          href="http://localhost:8501"
          target="_blank"
          rel="noreferrer"
          className="flex items-center justify-between px-3 py-2 rounded-md bg-zinc-900/60 border border-zinc-800 hover:border-zinc-700 text-xs font-mono text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          <span>Streamlit (Research)</span>
          <ExternalLink className="w-3.5 h-3.5" />
        </a>

        <div className="px-3 py-2 rounded bg-zinc-950/70 border border-zinc-900 text-[11px] font-mono text-zinc-500 space-y-1">
          <div className="flex justify-between">
            <span>PIPELINE:</span>
            <span className="text-zinc-300">LC-FIRST v1.0</span>
          </div>
          <div className="flex justify-between">
            <span>BENCHMARK:</span>
            <span className="text-zinc-300">106 OBJECTS</span>
          </div>
          <div
            className="flex justify-between items-center cursor-help"
            title="Real-ZTF Light-Curve Detector v2 has been evaluated on the frozen benchmark. Research-status novelty detection; not an astrophysical classification probability."
          >
            <span>ANOMALY GATE:</span>
            <span className="text-cyan-400 font-bold text-[10px]">REAL-ZTF RESEARCH</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
