"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  Download,
  FileText,
  ShieldAlert,
  ArrowLeft,
} from "lucide-react";
import dynamic from "next/dynamic";
import { fetchInvestigation, fetchLightCurve } from "@/lib/api";
import { InvestigationResult, LightCurvePoint } from "@/types/investigator";
import { LightCurveChart } from "@/components/investigation/LightCurveChart";
import { EmbeddingHeatmap } from "@/components/investigation/EmbeddingHeatmap";
import { EvidenceChain } from "@/components/investigation/EvidenceChain";

const EventCore = dynamic(() => import("@/components/three/EventCore"), {
  ssr: false,
  loading: () => <div className="h-64 rounded bg-[#0a0e17] border border-zinc-800 animate-pulse" />
});

const EmbeddingSpace = dynamic(() => import("@/components/three/EmbeddingSpace"), {
  ssr: false,
  loading: () => <div className="h-64 rounded bg-[#0a0e17] border border-zinc-800 animate-pulse" />
});

export default function InvestigationWorkstationPage() {
  const params = useParams();
  const candidateId = (params?.candidateId as string) || "CAND_SNIa_002";

  const [result, setResult] = useState<InvestigationResult | null>(null);
  const [lightcurvePoints, setLightcurvePoints] = useState<LightCurvePoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);
        const [invRes, lcRes] = await Promise.all([
          fetchInvestigation(candidateId),
          fetchLightCurve(candidateId).catch(() => ({ observations_count: 0, points: [] })),
        ]);
        setResult(invRes);
        setLightcurvePoints(lcRes.points || []);
      } catch (err: any) {
        setError(err.message || "Failed to load investigation data");
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [candidateId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
        <div className="text-sm font-mono text-zinc-400">
          Running end-to-end investigation on <span className="text-cyan-400 font-bold">{candidateId}</span>...
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="p-8 max-w-4xl mx-auto space-y-4">
        <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800 text-rose-300 font-mono text-xs">
          <b>Investigation Error:</b> {error || "Candidate could not be located in benchmark records."}
        </div>
        <Link href="/candidates" className="inline-flex items-center gap-2 text-cyan-400 font-mono text-xs hover:underline">
          <ArrowLeft className="w-3.5 h-3.5" /> Back to Candidate Registry
        </Link>
      </div>
    );
  }

  const char = result.event_characterization;
  const rep = result.representation_summary;
  const anom = result.anomaly_assessment;
  const meta = result.metadata;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8">
      {/* Workstation Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-zinc-800/80 pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Link href="/candidates" className="text-xs font-mono text-zinc-500 hover:text-cyan-400 flex items-center gap-1">
              <ArrowLeft className="w-3 h-3" /> Candidates
            </Link>
            <span className="text-zinc-600">/</span>
            <span className="text-xs font-mono text-cyan-400 uppercase font-semibold">{candidateId}</span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-bold font-mono text-white tracking-tight">
              {result.object_id}
            </h1>
            <span className="text-sm font-mono text-zinc-400">
              ({meta.ztf_object_id || meta.ztf_designation || candidateId})
            </span>
            <span className="px-2.5 py-0.5 rounded text-xs font-mono font-bold bg-cyan-950/80 text-cyan-300 border border-cyan-800/80">
              {meta.claimed_type || meta.class || "Unknown"}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-4 mt-2 text-xs font-mono text-zinc-400">
            <div>
              COORDS: <span className="text-zinc-200">{Number(meta.ra).toFixed(5)}°, {Number(meta.dec).toFixed(5)}°</span>
            </div>
            <div>
              RAW OBS: <span className="text-zinc-200">{result.provenance.raw_observation_count}</span>
            </div>
            <div>
              CLEAN OBS: <span className="text-zinc-200">{char.num_observations.value}</span>
            </div>
            <div>
              EXEC TIME: <span className="text-zinc-200">{result.execution_time_seconds.toFixed(3)} s</span>
            </div>
          </div>
        </div>

        {/* Action Controls & Report Downloads */}
        <div className="flex items-center gap-2">
          <a
            href={`/api/reports/${candidateId}/markdown`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-xs font-mono text-zinc-300 hover:text-white transition-colors"
          >
            <FileText className="w-3.5 h-3.5 text-zinc-400" />
            <span>Markdown Report</span>
          </a>
          <a
            href={`/api/reports/${candidateId}/json`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-xs font-mono text-zinc-300 hover:text-white transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-zinc-400" />
            <span>JSON Data</span>
          </a>
        </div>
      </div>

      {/* Signature Evidence Chain */}
      <EvidenceChain result={result} />

      {/* SECTION 02: LIGHT CURVE & OBSERVATION COVERAGE */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">02</span>
            <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
              Cleaned Photometric Light Curve
            </h2>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono text-zinc-400">
            <div>
              SPAN: <span className="text-zinc-200">{char.time_baseline_days.value} d</span>
            </div>
            <div>
              WINDOW: <span className="text-zinc-200">{char.window_duration_days.value} d</span>
            </div>
          </div>
        </div>

        {/* Observation Coverage Indicators */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-lg bg-[#0d1322] border border-zinc-800 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="text-zinc-400 uppercase">Filter Coverage:</span>
            {["zg", "zr", "zi", "V", "NIR"].map((band) => {
              const count = char.per_band_counts[band] || (band === "zg" ? char.per_band_counts["g"] : band === "zr" ? char.per_band_counts["r"] : band === "zi" ? char.per_band_counts["i"] : 0) || 0;
              const isAvailable = count > 0;
              return (
                <div
                  key={band}
                  className={`px-2 py-0.5 rounded text-[11px] font-bold border ${
                    isAvailable
                      ? "bg-emerald-950/60 text-emerald-400 border-emerald-800/80"
                      : "bg-zinc-950 text-zinc-600 border-zinc-800 line-through"
                  }`}
                >
                  {band}: {isAvailable ? `${count} obs` : "UNAVAILABLE"}
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-3 text-zinc-400">
            <div>
              TOKENS: <span className="text-emerald-400 font-bold">{char.valid_token_count.value}</span> / 50
            </div>
            <div>
              PADDING: <span className="text-zinc-300 font-bold">{((char.padding_fraction.value as number) * 100).toFixed(1)}%</span>
            </div>
            <div>
              CADENCE: <span className="text-zinc-300 font-bold">{char.cadence_median_days.value} d</span>
            </div>
          </div>
        </div>

        {/* Interactive Chart */}
        <LightCurveChart points={lightcurvePoints} candidateId={candidateId} />
      </div>

      {/* SECTION 03: NON-LEARNED EVENT CHARACTERIZATION */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">03</span>
            <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
              Non-Learned Event Characterization Matrix
            </h2>
          </div>
          <span className="text-[11px] font-mono text-zinc-500 italic">
            {char.classification_disclaimer}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* MEASURED FEATURES */}
          <div className="rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <span className="text-xs font-mono font-bold text-blue-400 uppercase tracking-wider">MEASURED</span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-blue-950 text-blue-400 border border-blue-800">Direct Photometry</span>
            </div>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Clean Observations</span>
                <span className="text-white font-bold">{char.num_observations.value}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Active Passbands</span>
                <span className="text-white font-bold">{char.num_filters.value} ({Object.keys(char.per_band_counts).join(", ")})</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Time Baseline</span>
                <span className="text-white font-bold">{char.time_baseline_days.value} d</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Peak Normalized Flux</span>
                <span className="text-white font-bold">{char.peak_normalized_flux.value}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-zinc-400">Minimum Normalized Flux</span>
                <span className="text-white font-bold">{char.minimum_normalized_flux.value}</span>
              </div>
            </div>
          </div>

          {/* DERIVED FEATURES */}
          <div className="rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider">DERIVED</span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">Statistical Metrics</span>
            </div>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Median Normalized Flux</span>
                <span className="text-white font-bold">{char.median_normalized_flux.value}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Flux Std Deviation</span>
                <span className="text-white font-bold">{char.flux_std.value}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Variability Amplitude</span>
                <span className="text-white font-bold">{char.variability_amplitude.value}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Median Flux Uncertainty</span>
                <span className="text-white font-bold">{char.median_flux_err.value}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-zinc-400">Median Cadence</span>
                <span className="text-white font-bold">{char.cadence_median_days.value} d</span>
              </div>
            </div>
          </div>

          {/* PROXY & UNAVAILABLE */}
          <div className="rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
              <span className="text-xs font-mono font-bold text-amber-400 uppercase tracking-wider">PROXY & GAPS</span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-amber-950 text-amber-400 border border-amber-800">Astrophysical Proxies</span>
            </div>
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Rise Time Proxy (Half-Max)</span>
                <span className="text-amber-300 font-bold">{char.rise_time_proxy_days.value} d</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Decline Time Proxy (Half-Max)</span>
                <span className="text-amber-300 font-bold">{char.decline_time_proxy_days.value} d</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-400">Rise/Decline Asymmetry</span>
                <span className="text-amber-300 font-bold">{char.rise_decline_asymmetry.value}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-zinc-900">
                <span className="text-zinc-500">Host Galaxy Offset</span>
                <span className="text-zinc-600 font-bold">UNAVAILABLE</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-zinc-500">Rest-Frame Absolute Mag</span>
                <span className="text-zinc-600 font-bold">UNAVAILABLE (No Redshift)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 04: REPRESENTATION (128-D HEATMAP & 3D VISUALIZATION) */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">04</span>
            <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
              Deep Light-Curve Representation & Subspace
            </h2>
          </div>
          <span className="text-xs font-mono text-zinc-500">
            Frozen 128-D Transformer Encoder Subspace
          </span>
        </div>

        {/* 3D Scientific Visualization Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <EventCore
            candidateId={candidateId}
            noveltyScore={anom.anomaly_score}
            isAnomaly={anom.anomaly_flag}
            tokenCount={rep.valid_token_count}
            embeddingNorm={rep.embedding_norm}
          />
          <EmbeddingSpace
            targetId={candidateId}
            targetEmbedding={rep.embedding}
          />
        </div>

        {/* 128-D Heatmap */}
        <EmbeddingHeatmap representation={rep} />
      </div>

      {/* SECTION 05: CONSERVATIVE SCIENTIFIC ANOMALY ASSESSMENT */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">05</span>
          <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
            Scientific Anomaly Assessment
          </h2>
        </div>

        <div className={`rounded-lg p-5 space-y-4 border-2 ${
          anom.anomaly_score_status === "REAL_ZTF_EVALUATED_RESEARCH"
            ? anom.anomaly_flag
              ? "bg-rose-950/20 border-rose-600/70"
              : "bg-emerald-950/20 border-emerald-700/60"
            : "bg-amber-950/20 border-amber-700/60"
        }`}>
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
            <div className="flex items-center gap-2.5">
              <ShieldAlert className={`w-5 h-5 ${
                anom.anomaly_flag ? "text-rose-400" : "text-emerald-400"
              }`} />
              <div>
                <div className="text-xs font-mono text-zinc-400 uppercase font-bold">EVALUATION STATUS:</div>
                <div className="text-lg font-mono font-black text-cyan-300 tracking-wide">
                  {anom.anomaly_score_status}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
              {anom.anomaly_score !== null && (
                <div className="bg-zinc-950/90 px-3 py-1.5 rounded border border-zinc-800 text-zinc-300">
                  NOVELTY SCORE: <span className="text-cyan-400 font-bold font-mono text-sm">{(anom.anomaly_score).toFixed(4)}</span>
                </div>
              )}
              {anom.anomaly_flag !== null && (
                <div className={`px-3 py-1.5 rounded font-bold border ${
                  anom.anomaly_flag
                    ? "bg-rose-950/80 text-rose-300 border-rose-800"
                    : "bg-emerald-950/80 text-emerald-300 border-emerald-800"
                }`}>
                  {anom.anomaly_flag ? "CANDIDATE ANOMALY" : "IN-DISTRIBUTION CANDIDATE"}
                </div>
              )}
              <div className="text-zinc-400 bg-zinc-950/80 px-3 py-1.5 rounded border border-zinc-800">
                DETECTOR: <span className="text-zinc-200">{anom.detector_source}</span>
              </div>
            </div>
          </div>

          <p className="text-xs font-mono text-zinc-300 leading-relaxed">
            {anom.domain_gap_notes}
          </p>

          <div className="space-y-1.5 pt-2 border-t border-zinc-800/80">
            <div className="text-[11px] font-mono text-cyan-400 font-bold uppercase tracking-wider">
              Documented Domain & Evaluation Constraints:
            </div>
            <ul className="space-y-1 text-xs font-mono text-zinc-400 list-disc list-inside">
              {anom.limitations.map((lim, idx) => (
                <li key={idx} className="leading-normal">{lim}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* SECTION 06: SCIENTIFIC HYPOTHESES */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">06</span>
            <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
              Candidate Transient Hypotheses
            </h2>
          </div>
          <span className="text-xs font-mono text-zinc-500">
            Ordered by investigator support from observed features and retrieved literature
          </span>
        </div>

        <div className="p-3 rounded-md bg-zinc-900/60 border border-zinc-800/80 text-[11px] font-mono text-zinc-400">
          <b className="text-zinc-300">Scientific Disclaimer:</b> ACEI has not performed a confirmed or validated astrophysical classification. Hypotheses represent provisional domain-grounded priors based on non-learned photometric heuristics and literature retrieval.
        </div>

        <div className="space-y-3">
          {result.candidate_hypotheses.map((hyp) => (
            <div
              key={hyp.rank}
              className={`p-4 rounded-lg bg-[#0a0e17] border ${
                hyp.rank === 1 ? "border-cyan-500/70 shadow-lg shadow-cyan-950/20" : "border-zinc-800"
              } space-y-3`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-800/80 pb-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className={`w-5 h-5 rounded flex items-center justify-center text-xs font-mono font-bold ${
                    hyp.rank === 1 ? "bg-cyan-500 text-black" : "bg-zinc-800 text-zinc-300"
                  }`}>
                    {hyp.rank}
                  </span>
                  <h3 className="text-sm font-mono font-bold text-white">
                    {hyp.name}
                  </h3>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-950/80 text-amber-300 border border-amber-800/80">
                    HYPOTHESIS — NOT CONFIRMED
                  </span>
                </div>

                <div className="text-xs font-mono text-zinc-400 sm:text-right">
                  {hyp.probability !== null ? (
                    <div>
                      <div>Relative Support: <b className="text-cyan-300">{hyp.probability.toFixed(2)}</b></div>
                      <div className="text-[10px] text-zinc-500">Relative support score; not a calibrated probability.</div>
                    </div>
                  ) : (
                    <span>Status: <b className="text-amber-400">{hyp.confidence_or_status}</b></span>
                  )}
                </div>
              </div>

              <p className="text-xs font-mono text-zinc-300 leading-relaxed">
                <b className="text-cyan-400">Hypothesis Rationale:</b> {hyp.justification}
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
                {/* Concept 1: Observed Features */}
                <div className="p-2.5 rounded bg-zinc-950/70 border border-zinc-900">
                  <span className="text-cyan-400 font-bold block mb-1 uppercase text-[10px]">Observed Candidate Features:</span>
                  <ul className="text-zinc-300 space-y-0.5 list-disc list-inside text-[11px]">
                    <li>Rise Proxy: <code>{char.rise_time_proxy_days.value} d</code></li>
                    <li>Decline Proxy: <code>{char.decline_time_proxy_days.value} d</code></li>
                    <li>Asymmetry: <code>{char.rise_decline_asymmetry.value}</code></li>
                    <li>Active Bands: <code>{Object.keys(char.per_band_counts).join(", ")}</code></li>
                  </ul>
                </div>

                {/* Concept 2: Supporting Literature Evidence */}
                <div className="p-2.5 rounded bg-zinc-950/70 border border-zinc-900">
                  <span className="text-emerald-400 font-bold block mb-1 uppercase text-[10px]">Supporting Literature Evidence:</span>
                  <ul className="text-zinc-400 space-y-0.5 list-disc list-inside text-[11px]">
                    {hyp.supporting_evidence.length > 0 ? (
                      hyp.supporting_evidence.map((s, idx) => <li key={idx}><code>{s}</code></li>)
                    ) : (
                      <li>None cited</li>
                    )}
                  </ul>
                </div>

                {/* Concept 3: Missing / Unmeasured Evidence */}
                <div className="p-2.5 rounded bg-zinc-950/70 border border-zinc-900">
                  <span className="text-amber-400 font-bold block mb-1 uppercase text-[10px]">Missing / Unmeasured Evidence:</span>
                  <ul className="text-zinc-400 space-y-0.5 list-disc list-inside text-[11px]">
                    {hyp.contradictory_evidence.length > 0 ? (
                      hyp.contradictory_evidence.map((c, idx) => <li key={idx}>{c}</li>)
                    ) : (
                      <li>No unmeasured gap noted</li>
                    )}
                  </ul>
                </div>
              </div>

              <div className="text-xs font-mono text-zinc-400 pt-1">
                <b className="text-zinc-200">Distinguishing Diagnostic Test:</b> {hyp.distinguishing_criteria}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* SECTION 07: KNOWLEDGE BASE EVIDENCE */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">07</span>
            <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
              Astrophysical Literature & RAG Evidence
            </h2>
          </div>
          <span className="text-xs font-mono text-zinc-500">
            Retrieved from In-Memory Vector Store
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {result.evidence.map((ev) => (
            <div key={ev.doc_id} className="p-4 rounded-lg bg-[#0a0e17] border border-zinc-800 space-y-2 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500">
                  <span>{ev.doc_id}</span>
                  <span className="text-cyan-400 font-semibold">Sim: {ev.relevance_score.toFixed(4)}</span>
                </div>
                <h4 className="text-xs font-mono font-bold text-white mt-1 line-clamp-2">
                  {ev.title}
                </h4>
                <p className="text-xs text-zinc-400 mt-2 line-clamp-4 leading-relaxed font-sans italic border-l-2 border-cyan-800 pl-2">
                  "{ev.snippet}"
                </p>
              </div>

              <div className="text-[10px] font-mono text-zinc-600 truncate pt-2 border-t border-zinc-900">
                URI: {ev.source_uri}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* SECTION 08: OBSERVATION PLANNER */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">08</span>
            <h2 className="text-sm font-bold font-mono tracking-wider text-zinc-200 uppercase">
              Follow-Up Observation Planner
            </h2>
          </div>
          <span className="text-xs font-mono text-zinc-500">
            Target-of-Opportunity Decision Recommendations
          </span>
        </div>

        <div className="space-y-3">
          {result.recommended_observations.map((rec) => (
            <div
              key={rec.priority}
              className="p-4 rounded-lg bg-[#0d1322] border border-zinc-800 flex flex-col md:flex-row md:items-center justify-between gap-4"
            >
              <div className="space-y-1.5 flex-1">
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                    rec.urgency === "HIGH" ? "bg-rose-950 text-rose-400 border border-rose-800" : "bg-zinc-900 text-zinc-300 border border-zinc-800"
                  }`}>
                    PRIORITY {rec.priority} ({rec.urgency})
                  </span>
                  <span className="text-xs font-mono font-bold text-white">{rec.action}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 text-amber-400 border border-zinc-800">
                    RECOMMENDED
                  </span>
                </div>
                <p className="text-xs font-mono text-zinc-300">
                  <b>Scientific Rationale:</b> {rec.scientific_rationale}
                </p>
                <p className="text-xs font-mono text-zinc-400">
                  <b>Information Gap:</b> {rec.information_gap}
                </p>
              </div>

              <div className="text-right shrink-0">
                <div className="text-[11px] font-mono text-zinc-500">TARGET PASSBAND</div>
                <div className="text-xs font-mono font-bold text-cyan-400 mt-0.5 uppercase">
                  {rec.target_band || "ANY"}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
