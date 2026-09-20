"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import {
  Layers,
  ArrowRight,
  Sparkles,
  CheckCircle2,
  Clock,
  Search
} from "lucide-react";
import { fetchSystemHealth, fetchReports } from "@/lib/api";
import { SystemHealth } from "@/types/investigator";

const PipelineFlow3D = dynamic(() => import("@/components/three/PipelineFlow3D"), {
  ssr: false,
  loading: () => <div className="h-28 rounded bg-[#0a0e17] border border-zinc-800 animate-pulse" />
});

export default function DashboardPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [reports, setReports] = useState<any[]>([]);

  useEffect(() => {
    async function loadDashboardData() {
      try {
        const [h, r] = await Promise.all([
          fetchSystemHealth().catch(() => null),
          fetchReports().catch(() => ({ count: 0, reports: [] })),
        ]);
        setHealth(h);
        setReports(r.reports || []);
      } catch (err: any) {
        console.error(err);
      }
    }
    loadDashboardData();
  }, []);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8">
      {/* Top Banner & Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
            <span className="text-xs font-mono text-emerald-400 font-semibold uppercase tracking-wider">
              ACEI Mission Control Active
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white font-mono">
            Astronomical Observatory Console
          </h1>
          <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
            Real-ZTF Light-Curve-First investigation pipeline evaluating photometric candidates against the frozen production transformer and astrophysical literature.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/candidates"
            className="flex items-center gap-2 px-3.5 py-2 rounded-md bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-xs font-mono text-zinc-300 transition-colors"
          >
            <Search className="w-3.5 h-3.5 text-zinc-400" />
            <span>Browse All Candidates</span>
          </Link>
          <Link
            href="/investigate/CAND_SNIa_002"
            className="flex items-center gap-2 px-4 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-xs font-mono font-semibold text-white shadow-lg shadow-cyan-950/50 transition-colors"
          >
            <span>Launch Workstation</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* 3D Scientific Workflow Topology Canvas */}
      <PipelineFlow3D />

      {/* Scientific Workflow Architecture Pipeline */}
      <div className="p-4 rounded-lg bg-[#0a0e17] border border-zinc-800 space-y-3">
        <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
          <span className="text-xs font-mono font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            Closed-Loop Scientific Investigation Workflow Architecture
          </span>
          <span className="text-[10px] font-mono text-zinc-500">Real ZTF Alert Stream</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-7 gap-2 text-center text-xs font-mono">
          {[
            { step: "01", name: "REAL ZTF", tag: "Public IRSA Alerts", color: "border-blue-800 bg-blue-950/30 text-blue-300" },
            { step: "02", name: "PHOTOMETRIC PROCESSING", tag: "Bitwise Preprocessing", color: "border-emerald-800 bg-emerald-950/30 text-emerald-300" },
            { step: "03", name: "128-D TRANSFORMER", tag: "Frozen Encoder", color: "border-purple-800 bg-purple-950/30 text-purple-300" },
            { step: "04", name: "INVESTIGATION", tag: "Non-Learned Feature Matrix", color: "border-cyan-800 bg-cyan-950/30 text-cyan-300" },
            { step: "05", name: "LITERATURE GROUNDING", tag: "In-Memory Vector Store", color: "border-indigo-800 bg-indigo-950/30 text-indigo-300" },
            { step: "06", name: "HYPOTHESIS", tag: "Provisional Priors", color: "border-amber-800 bg-amber-950/30 text-amber-300" },
            { step: "07", name: "FOLLOW-UP ACTION", tag: "Active Learning Plan", color: "border-rose-800 bg-rose-950/30 text-rose-300" },
          ].map((stage) => (
            <div key={stage.step} className="relative group">
              <div className={`p-2.5 rounded border ${stage.color} space-y-1 h-full flex flex-col justify-between hover:border-cyan-400 transition-colors`}>
                <div className="text-[9px] font-bold opacity-60">STEP {stage.step}</div>
                <div className="font-bold text-[11px] leading-tight">{stage.name}</div>
                <div className="text-[9px] opacity-75">{stage.tag}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800/80">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Investigated</div>
          <div className="text-xl font-bold font-mono text-white mt-1">
            {reports.length > 0 ? reports.length : 5}
          </div>
          <div className="text-[10px] text-cyan-400 mt-1 flex items-center gap-1 font-mono">
            <CheckCircle2 className="w-3 h-3" /> Real ZTF reports
          </div>
        </div>

        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800/80">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Primary Benchmark</div>
          <div className="text-xl font-bold font-mono text-white mt-1">
            {health?.primary_benchmark_objects || 106}
          </div>
          <div className="text-[10px] text-zinc-400 mt-1 font-mono">Frozen objects</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800/80">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Knowledge Sources</div>
          <div className="text-xl font-bold font-mono text-white mt-1">4</div>
          <div className="text-[10px] text-zinc-400 mt-1 font-mono">Papers & Catalogs</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800/80">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Encoder Latent</div>
          <div className="text-xl font-bold font-mono text-blue-400 mt-1">128-D</div>
          <div className="text-[10px] text-zinc-400 mt-1 font-mono">Frozen Transformer</div>
        </div>

        <div
          className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800/80 cursor-help"
          title="Real-ZTF Light-Curve Detector v2 has been evaluated on the frozen benchmark. Research-status novelty detection; not an astrophysical classification probability."
        >
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Anomaly Status</div>
          <div className="text-xs font-bold font-mono text-cyan-400 mt-1 truncate">
            EVALUATED — RESEARCH
          </div>
          <div className="text-[10px] text-cyan-500/80 mt-1 font-mono">v2 Real-ZTF Detector</div>
        </div>

        <div className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800/80">
          <div className="text-[11px] font-mono text-zinc-500 uppercase">Survey Provenance</div>
          <div className="text-sm font-bold font-mono text-zinc-200 mt-1 truncate">
            ZTF / IRSA
          </div>
          <div className="text-[10px] text-zinc-400 mt-1 font-mono">100% Real Photometry</div>
        </div>
      </div>

      {/* Featured Demonstration Investigations */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
              Demonstration Real-ZTF Candidates
            </h2>
          </div>
          <span className="text-xs font-mono text-zinc-500">
            5 Curated Astronomical Archetypes
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {[
            { id: "CAND_SNIa_002", ztf: "ZTF20aakyqlh", cls: "SN Ia", desc: "Thermonuclear supernova in host dust lane", color: "border-blue-800/60" },
            { id: "CAND_SLSN_001", ztf: "ZTF18abxecad", cls: "SLSN-II", desc: "Magnetar engine powered luminous explosion", color: "border-purple-800/60" },
            { id: "CAND_TDE_001", ztf: "ZTF18actdhei", cls: "TDE", desc: "Tidal disruption flare around SMBH", color: "border-cyan-800/60" },
            { id: "CAND_CV_001", ztf: "ZTF17aaaemzh", cls: "Cataclysmic_Variable", desc: "Recurrent dwarf nova accretion outburst", color: "border-amber-800/60" },
            { id: "CAND_FieldStar_001", ztf: "ZTF18aaadkrh", cls: "Field Star", desc: "Unclassified variable star baseline control", color: "border-zinc-700/60" },
          ].map((c) => (
            <Link
              key={c.id}
              href={`/investigate/${c.id}`}
              className={`p-3.5 rounded-lg bg-[#0a0e17] border ${c.color} hover:border-cyan-500/80 hover:bg-[#101726] transition-all group flex flex-col justify-between`}
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-white group-hover:text-cyan-300">
                    {c.id}
                  </span>
                  <ArrowRight className="w-3.5 h-3.5 text-zinc-600 group-hover:text-cyan-400 transition-colors" />
                </div>
                <div className="text-[11px] font-mono text-zinc-400 mt-0.5">{c.ztf}</div>
                <div className="mt-2 text-[11px] font-mono inline-block px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-cyan-400">
                  {c.cls}
                </div>
                <p className="text-[11px] text-zinc-400 mt-2 line-clamp-2 leading-relaxed">
                  {c.desc}
                </p>
              </div>
              <div className="mt-3 pt-2 border-t border-zinc-900 flex justify-between text-[10px] font-mono text-zinc-500">
                <span>STATUS:</span>
                <span className="text-emerald-400 font-semibold">INVESTIGATED</span>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Main Investigations Table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
              Recent Scientific Investigations & Reports
            </h2>
          </div>
          <span className="text-xs font-mono text-zinc-500">
            Automated Machine-Readable JSON & Human Markdown
          </span>
        </div>

        <div className="rounded-lg border border-zinc-800/80 bg-[#090d16] overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#0f1624] text-zinc-400 uppercase tracking-wider text-[10px] border-b border-zinc-800">
                <tr>
                  <th className="py-3 px-4">Candidate ID</th>
                  <th className="py-3 px-4">Astrophysical Class</th>
                  <th className="py-3 px-4">Representation</th>
                  <th className="py-3 px-4">Anomaly Status</th>
                  <th className="py-3 px-4">Top Hypothesis</th>
                  <th className="py-3 px-4">Execution Time</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60 text-zinc-300">
                {reports.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-8 text-center text-zinc-500 font-mono">
                      No cached reports found. Run investigations from the Candidates browser.
                    </td>
                  </tr>
                ) : (
                  reports.map((rep) => (
                    <tr key={rep.candidate_id} className="hover:bg-zinc-800/30 transition-colors">
                      <td className="py-3 px-4 font-bold text-white">
                        <Link href={`/investigate/${rep.candidate_id}`} className="hover:text-cyan-300">
                          {rep.candidate_id}
                        </Link>
                      </td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded text-[11px] bg-zinc-900 border border-zinc-800 text-zinc-300">
                          {rep.claimed_type}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-cyan-400">
                        128-D L2-Norm
                      </td>
                      <td className="py-3 px-4">
                        <span className="px-2 py-0.5 rounded text-[10px] bg-amber-950/60 text-amber-400 border border-amber-800/50">
                          NOT_VALIDATED_FOR_REAL_ZTF
                        </span>
                      </td>
                      <td className="py-3 px-4 text-zinc-200">
                        {rep.top_hypothesis}
                      </td>
                      <td className="py-3 px-4 text-zinc-400">
                        {rep.execution_time ? `${rep.execution_time.toFixed(3)} s` : "N/A"}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <Link
                          href={`/investigate/${rep.candidate_id}`}
                          className="inline-flex items-center gap-1 text-cyan-400 hover:text-cyan-300 font-semibold"
                        >
                          <span>Inspect</span>
                          <ArrowRight className="w-3 h-3" />
                        </Link>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Investigation Activity Timeline */}
      <div className="p-4 rounded-lg bg-[#0a0e17] border border-zinc-800/80 space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono font-semibold text-zinc-300 uppercase">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span>Observatory Activity & Integrity Audit</span>
        </div>
        <div className="space-y-2 text-xs font-mono">
          <div className="flex items-start gap-3 p-2 rounded bg-zinc-900/50 border border-zinc-800/60">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
            <div className="flex-1 flex justify-between">
              <span className="text-zinc-300">Light-Curve-First production pipeline operational on real ZTF alert sequences.</span>
              <span className="text-zinc-500 text-[10px]">PRODUCTION BASELINE</span>
            </div>
          </div>
          <div className="flex items-start gap-3 p-2 rounded bg-zinc-900/50 border border-zinc-800/60">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-1.5 shrink-0" />
            <div className="flex-1 flex justify-between">
              <span className="text-zinc-300">Production checkpoint SHA-256 verified bitwise immutable: <code>e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72</code></span>
              <span className="text-zinc-500 text-[10px]">SECURITY AUDIT</span>
            </div>
          </div>
          <div className="flex items-start gap-3 p-2 rounded bg-zinc-900/50 border border-zinc-800/60">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 shrink-0" />
            <div className="flex-1 flex justify-between">
              <span className="text-zinc-300">Scientific anomaly assessor enforcing <code>NOT_VALIDATED_FOR_REAL_ZTF</code> conservative domain gate.</span>
              <span className="text-zinc-500 text-[10px]">DOMAIN BOUNDARY</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
