"use client";

import React, { useState } from "react";
import { Telescope, Eye, BookOpen, Sparkles, CalendarClock, ChevronDown, ChevronUp } from "lucide-react";
import { InvestigationResult } from "@/types/investigator";

interface EvidenceChainProps {
  result: InvestigationResult;
}

export function EvidenceChain({ result }: EvidenceChainProps) {
  const [activeStep, setActiveStep] = useState<string | null>("03");
  const char = result.event_characterization;
  const topHyp = result.candidate_hypotheses[0];
  const topEv = result.evidence[0];
  const topRec = result.recommended_observations[0];

  const steps = [
    {
      step: "01",
      title: "OBSERVATION",
      subtitle: `${char.num_observations.value || 0} epochs across ${char.num_filters.value || 0} bands`,
      tag: "REAL ZTF PHOTOMETRY",
      icon: Eye,
      color: "border-blue-800/80 bg-blue-950/20 text-blue-400",
    },
    {
      step: "02",
      title: "FEATURE EXTRACTION",
      subtitle: `Rise ~${char.rise_time_proxy_days.value || "-"}d | Amp ~${char.variability_amplitude.value || "-"}`,
      tag: "NON-LEARNED METRICS",
      icon: Sparkles,
      color: "border-emerald-800/80 bg-emerald-950/20 text-emerald-400",
    },
    {
      step: "03",
      title: "HYPOTHESIS PRIOR",
      subtitle: topHyp?.name || "Unclassified Transient",
      tag: "PROVISIONAL PRIOR",
      icon: Telescope,
      color: "border-purple-800/80 bg-purple-950/20 text-purple-400",
    },
    {
      step: "04",
      title: "LITERATURE EVIDENCE",
      subtitle: topEv ? `[${topEv.doc_id}] ${topEv.title.slice(0, 30)}...` : "Knowledge base passages",
      tag: "IN-MEMORY RAG RETRIEVAL",
      icon: BookOpen,
      color: "border-cyan-800/80 bg-cyan-950/20 text-cyan-400",
    },
    {
      step: "05",
      title: "FOLLOW-UP ACTION",
      subtitle: topRec?.action || "Optical spectroscopy",
      tag: "ACTIVE LEARNING PLAN",
      icon: CalendarClock,
      color: "border-amber-800/80 bg-amber-950/20 text-amber-400",
    },
  ];

  const toggleStep = (step: string) => {
    setActiveStep(activeStep === step ? null : step);
  };

  return (
    <div className="rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800/80 pb-3">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-xs font-mono font-bold text-white uppercase tracking-wider">
            Signature Scientific Evidence Chain
          </span>
        </div>
        <span className="text-[11px] font-mono text-zinc-500">
          Click stage to inspect underlying data
        </span>
      </div>

      {/* Steps Row */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-2 relative">
        {steps.map((st) => {
          const Icon = st.icon;
          const isSelected = activeStep === st.step;
          return (
            <div
              key={st.step}
              onClick={() => toggleStep(st.step)}
              className={`p-3 rounded-md border ${st.color} cursor-pointer transition-all flex flex-col justify-between relative group ${
                isSelected ? "ring-1 ring-cyan-400 shadow-md shadow-cyan-950/40" : "hover:border-cyan-500/60"
              }`}
            >
              <div>
                <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500 mb-1">
                  <span>STEP {st.step}</span>
                  <Icon className="w-3.5 h-3.5 text-zinc-400 group-hover:text-cyan-400 transition-colors" />
                </div>
                <div className="text-xs font-mono font-bold text-white uppercase truncate">
                  {st.title}
                </div>
                <div className="text-[11px] text-zinc-300 mt-1 line-clamp-2 leading-snug">
                  {st.subtitle}
                </div>
              </div>

              <div className="mt-3 pt-1.5 border-t border-zinc-800/60 flex items-center justify-between text-[9px] font-mono font-semibold tracking-wider uppercase">
                <span className="text-zinc-400 truncate">{st.tag}</span>
                {isSelected ? <ChevronUp className="w-3 h-3 text-cyan-400" /> : <ChevronDown className="w-3 h-3 text-zinc-600" />}
              </div>
            </div>
          );
        })}
      </div>

      {/* Interactive Expandable Stage Inspector */}
      {activeStep && (
        <div className="p-4 rounded-md bg-[#060910] border border-zinc-800/90 space-y-3 font-mono text-xs">
          {activeStep === "01" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-blue-400 font-bold border-b border-zinc-800 pb-1 uppercase">
                <span>Stage 01 Audit: Photometric Observation Data</span>
                <span>ZTF / NASA IPAC</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-zinc-300 pt-1">
                <div>Raw Epochs: <b className="text-white">{result.provenance.raw_observation_count}</b></div>
                <div>Clean Epochs: <b className="text-white">{char.num_observations.value}</b></div>
                <div>Time Baseline: <b className="text-white">{char.time_baseline_days.value} days</b></div>
                <div>Passbands: <b className="text-emerald-400">{Object.keys(char.per_band_counts).join(", ")}</b></div>
              </div>
            </div>
          )}

          {activeStep === "02" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-emerald-400 font-bold border-b border-zinc-800 pb-1 uppercase">
                <span>Stage 02 Audit: Non-Learned Physical Feature Proxies</span>
                <span>Direct Feature Extraction</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-zinc-300 pt-1">
                <div>Rise Time Proxy: <b className="text-amber-300">{char.rise_time_proxy_days.value} d</b></div>
                <div>Decline Time Proxy: <b className="text-amber-300">{char.decline_time_proxy_days.value} d</b></div>
                <div>Asymmetry Ratio: <b className="text-white">{char.rise_decline_asymmetry.value}</b></div>
                <div>Variability Amplitude: <b className="text-white">{char.variability_amplitude.value}</b></div>
              </div>
            </div>
          )}

          {activeStep === "03" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-purple-400 font-bold border-b border-zinc-800 pb-1 uppercase">
                <span>Stage 03 Audit: Provisional Scientific Hypotheses</span>
                <span>Not Confirmed</span>
              </div>
              <div className="space-y-2 pt-1">
                {result.candidate_hypotheses.map((h) => (
                  <div key={h.rank} className="p-2 rounded bg-zinc-900/80 border border-zinc-800 flex justify-between items-start">
                    <div>
                      <span className="font-bold text-white">[{h.rank}] {h.name}</span>
                      <p className="text-[11px] text-zinc-400 mt-0.5">{h.justification}</p>
                    </div>
                    {h.probability !== null && (
                      <span className="text-cyan-300 font-bold ml-2 shrink-0">Support: {h.probability.toFixed(2)}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeStep === "04" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-cyan-400 font-bold border-b border-zinc-800 pb-1 uppercase">
                <span>Stage 04 Audit: RAG Literature Evidence Passages</span>
                <span>In-Memory Vector Store</span>
              </div>
              <div className="space-y-2 pt-1">
                {result.evidence.map((ev) => (
                  <div key={ev.doc_id} className="p-2 rounded bg-zinc-900/80 border border-zinc-800 space-y-1">
                    <div className="flex justify-between text-[11px]">
                      <span className="font-bold text-white">{ev.doc_id}: {ev.title}</span>
                      <span className="text-cyan-400">Sim: {ev.relevance_score.toFixed(4)}</span>
                    </div>
                    <p className="text-[11px] text-zinc-400 italic">"{ev.snippet}"</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeStep === "05" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-amber-400 font-bold border-b border-zinc-800 pb-1 uppercase">
                <span>Stage 05 Audit: Active Learning Follow-Up Recommendations</span>
                <span>Decision-Theoretic Planner</span>
              </div>
              <div className="space-y-2 pt-1">
                {result.recommended_observations.map((rec) => (
                  <div key={rec.priority} className="p-2 rounded bg-zinc-900/80 border border-zinc-800 flex justify-between items-start">
                    <div>
                      <span className="font-bold text-white">Priority {rec.priority} ({rec.urgency}): {rec.action}</span>
                      <p className="text-[11px] text-zinc-400 mt-0.5"><b>Rationale:</b> {rec.scientific_rationale}</p>
                    </div>
                    <span className="text-amber-400 font-bold text-[10px] ml-2 shrink-0 uppercase">{rec.target_band || "ANY"}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
