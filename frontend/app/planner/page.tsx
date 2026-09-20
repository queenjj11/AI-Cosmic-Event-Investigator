"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { AlertCircle, ArrowRight } from "lucide-react";
import { fetchReports, fetchInvestigation } from "@/lib/api";
import { InvestigationResult, ObservationRecommendation } from "@/types/investigator";

const FollowupTarget3D = dynamic(() => import("@/components/three/FollowupTarget3D"), {
  ssr: false,
  loading: () => <div className="h-44 rounded bg-[#0a0e17] border border-zinc-800 animate-pulse" />
});

interface PlanItem {
  candidateId: string;
  candidateClass: string;
  recommendation: ObservationRecommendation;
}

export default function PlannerPage() {
  const [plans, setPlans] = useState<PlanItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedUrgency, setSelectedUrgency] = useState<string>("ALL");

  useEffect(() => {
    async function loadAllPlans() {
      try {
        setLoading(true);
        const repRes = await fetchReports();
        const candidateIds = repRes.reports.map((r: any) => r.candidate_id);

        const items: PlanItem[] = [];
        for (const cid of candidateIds) {
          try {
            const inv: InvestigationResult = await fetchInvestigation(cid);
            inv.recommended_observations.forEach((rec) => {
              items.push({
                candidateId: cid,
                candidateClass: inv.metadata.claimed_type || inv.metadata.class || "Unknown",
                recommendation: rec,
              });
            });
          } catch (e) {
            console.error(e);
          }
        }
        setPlans(items);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadAllPlans();
  }, []);

  const filteredPlans = plans.filter((p) => {
    return selectedUrgency === "ALL" || p.recommendation.urgency === selectedUrgency;
  });

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-mono text-cyan-400 font-semibold uppercase tracking-wider">
            Active Learning Decision Engine
          </span>
        </div>
        <h1 className="text-2xl font-bold font-mono text-white">Target-of-Opportunity Observation Planner</h1>
        <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
          Decision-theoretic follow-up queue maximizing information gain to resolve transient classification ambiguity.
        </p>
      </div>

      {/* Scientific Operational Policy Disclaimer Banner */}
      <div className="p-4 rounded-lg bg-zinc-900/60 border border-zinc-800 text-xs font-mono text-zinc-400 flex items-start gap-3">
        <AlertCircle className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <span>
          <b className="text-zinc-200">Scientific Operational Policy:</b> Recommendations are decision-theoretic recommendations based on identified photometric coverage gaps and hypothesis disambiguation criteria. Recommendations do not represent actual telescope scheduling, API robotic booking, or assert instrument availability.
        </span>
      </div>

      {/* 3D Follow-up Observation Target Model */}
      <FollowupTarget3D />

      {/* Filter Bar */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-[#0d1322] border border-zinc-800">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-zinc-400">Urgency Filter:</span>
          {["ALL", "HIGH", "NORMAL", "LOW"].map((urg) => (
            <button
              key={urg}
              onClick={() => setSelectedUrgency(urg)}
              className={`px-3 py-1 rounded text-xs font-mono transition-all border ${
                selectedUrgency === urg
                  ? "bg-cyan-950 border-cyan-800 text-cyan-300 font-bold"
                  : "bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-white"
              }`}
            >
              {urg}
            </button>
          ))}
        </div>

        <div className="text-xs font-mono text-zinc-400">
          Total Decision Plans: <span className="text-white font-bold">{filteredPlans.length}</span>
        </div>
      </div>

      {/* Recommendations Queue */}
      <div className="space-y-4">
        {loading ? (
          <div className="p-12 text-center text-zinc-500 font-mono text-xs">
            Synthesizing observation plans across active investigations...
          </div>
        ) : filteredPlans.length === 0 ? (
          <div className="p-12 text-center text-zinc-500 font-mono text-xs">
            No recommendations matching selected urgency criteria.
          </div>
        ) : (
          filteredPlans.map((item, idx) => (
            <div
              key={idx}
              className="p-4 rounded-lg bg-[#0a0e17] border border-zinc-800 space-y-3 hover:border-cyan-500/60 transition-colors"
            >
              {/* Header row */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800/80 pb-2">
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                    item.recommendation.urgency === "HIGH"
                      ? "bg-rose-950/80 text-rose-400 border border-rose-800"
                      : "bg-zinc-900 text-zinc-300 border border-zinc-800"
                  }`}>
                    PRIORITY {item.recommendation.priority} ({item.recommendation.urgency})
                  </span>
                  <span className="text-xs font-mono font-bold text-white">
                    Target: <Link href={`/investigate/${item.candidateId}`} className="text-cyan-300 hover:underline">{item.candidateId}</Link> ({item.candidateClass})
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-amber-950/60 text-amber-300 border border-amber-800/60 font-bold">
                    RECOMMENDED
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono text-zinc-400">
                    PASSBAND: <b className="text-cyan-400 uppercase">{item.recommendation.target_band || "ANY"}</b>
                  </span>
                  <Link
                    href={`/investigate/${item.candidateId}`}
                    className="flex items-center gap-1 text-xs font-mono text-cyan-400 hover:text-cyan-300 font-semibold"
                  >
                    <span>Workstation</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                </div>
              </div>

              {/* 4-Stage Decision Flow */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded bg-[#060910] border border-zinc-800/80 space-y-1">
                  <span className="text-[10px] text-zinc-500 uppercase font-bold block">1. Information Gap</span>
                  <p className="text-zinc-300 leading-relaxed">{item.recommendation.information_gap}</p>
                </div>

                <div className="p-2.5 rounded bg-[#060910] border border-zinc-800/80 space-y-1">
                  <span className="text-[10px] text-cyan-400 uppercase font-bold block">2. Proposed Observation</span>
                  <p className="text-white font-bold leading-relaxed">{item.recommendation.action}</p>
                </div>

                <div className="p-2.5 rounded bg-[#060910] border border-zinc-800/80 space-y-1">
                  <span className="text-[10px] text-emerald-400 uppercase font-bold block">3. Scientific Purpose</span>
                  <p className="text-zinc-300 leading-relaxed">{item.recommendation.scientific_rationale}</p>
                </div>

                <div className="p-2.5 rounded bg-[#060910] border border-zinc-800/80 space-y-1">
                  <span className="text-[10px] text-amber-400 uppercase font-bold block">4. Expected Info Gain</span>
                  <p className="text-amber-200 leading-relaxed">
                    Maximizes Shannon information entropy reduction to resolve top provisional hypothesis ambiguity.
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
