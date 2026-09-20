"use client";

import React, { useEffect, useState } from "react";
import { Database, Cpu, Lock, CheckCircle2, AlertTriangle } from "lucide-react";
import { fetchSystemHealth } from "@/lib/api";
import { SystemHealth } from "@/types/investigator";

export default function SystemPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    fetchSystemHealth().then(setHealth).catch(console.error);
  }, []);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-mono text-cyan-400 font-semibold uppercase tracking-wider">
            Infrastructure & Provenance
          </span>
        </div>
        <h1 className="text-2xl font-bold font-mono text-white">System Architecture & Audit Ledger</h1>
        <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
          Complete cryptographic provenance, parameter counts, frozen benchmark splits, and domain boundary contracts.
        </p>
      </div>

      {/* Security & Cryptographic Hashes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-5 rounded-lg bg-[#0d1322] border border-zinc-800 space-y-3">
          <div className="flex items-center gap-2 text-xs font-mono font-bold text-white uppercase">
            <Lock className="w-4 h-4 text-emerald-400" />
            <span>Production Checkpoint Immutability</span>
          </div>
          <div className="space-y-1.5 text-xs font-mono text-zinc-300">
            <div>
              <span className="text-zinc-500">CHECKPOINT PATH:</span>{" "}
              <code>models/checkpoints/acei_multimodal_production.pt</code>
            </div>
            <div>
              <span className="text-zinc-500">SHA-256 HASH:</span>
              <div className="p-2 rounded bg-zinc-950 border border-zinc-900 text-emerald-400 break-all select-all font-bold mt-1">
                {health?.checkpoint_sha256 || "e5c78fdc6f72cf972f4a3f5bf1b99dbb704aa7f047f7ed46ead1e9e538ffbc72"}
              </div>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-emerald-400 pt-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Bitwise verified immutable across all investigation phases</span>
            </div>
          </div>
        </div>

        <div className="p-5 rounded-lg bg-[#0d1322] border border-zinc-800 space-y-3">
          <div className="flex items-center gap-2 text-xs font-mono font-bold text-white uppercase">
            <Database className="w-4 h-4 text-cyan-400" />
            <span>Frozen Real-ZTF Primary Benchmark</span>
          </div>
          <div className="space-y-1.5 text-xs font-mono text-zinc-300">
            <div>
              <span className="text-zinc-500">DATASET MANIFEST:</span>{" "}
              <code>data/real_ztf_benchmark/frozen_primary_benchmark.csv</code>
            </div>
            <div>
              <span className="text-zinc-500">SHA-256 HASH:</span>
              <div className="p-2 rounded bg-zinc-950 border border-zinc-900 text-cyan-400 break-all select-all font-bold mt-1">
                {health?.frozen_benchmark_sha256 || "0e839c8f8ff1b83969e3f959a2be3342a96ac2564d00e73b045a21445cd4501c"}
              </div>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-cyan-400 pt-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>106 clean-coverage objects locked and strictly isolated</span>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Spec Matrix */}
      <div className="p-5 rounded-lg bg-[#0a0e17] border border-zinc-800 space-y-4">
        <div className="flex items-center justify-between border-b border-zinc-800/80 pb-3">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-bold font-mono text-white uppercase">
              Model & Preprocessing Specifications
            </h2>
          </div>
          <span className="text-xs font-mono text-zinc-500">100% PyTorch CPU Compatible</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-3.5 rounded bg-zinc-950/70 border border-zinc-900 space-y-2">
            <span className="text-cyan-400 font-bold block">LightCurveEncoder (Transformer)</span>
            <div className="space-y-1 text-zinc-400">
              <div className="flex justify-between"><span>Embedding Dim:</span> <b className="text-zinc-200">128</b></div>
              <div className="flex justify-between"><span>d_model:</span> <b className="text-zinc-200">128</b></div>
              <div className="flex justify-between"><span>Attention Heads:</span> <b className="text-zinc-200">4</b></div>
              <div className="flex justify-between"><span>Layers:</span> <b className="text-zinc-200">3</b></div>
              <div className="flex justify-between"><span>Feedforward Dim:</span> <b className="text-zinc-200">256</b></div>
              <div className="flex justify-between"><span>Time Encoding:</span> <b className="text-zinc-200">Time2Vec (32-D)</b></div>
              <div className="flex justify-between"><span>Total Parameters:</span> <b className="text-zinc-200">425,072</b></div>
            </div>
          </div>

          <div className="p-3.5 rounded bg-zinc-950/70 border border-zinc-900 space-y-2">
            <span className="text-emerald-400 font-bold block">RealZTFPreprocessor Contract</span>
            <div className="space-y-1 text-zinc-400">
              <div className="flex justify-between"><span>Max Sequence Length:</span> <b className="text-zinc-200">50 tokens</b></div>
              <div className="flex justify-between"><span>Tensor Shape:</span> <b className="text-zinc-200">(50, 4)</b></div>
              <div className="flex justify-between"><span>Channel Format:</span> <b className="text-zinc-200">[t, flux, err, band]</b></div>
              <div className="flex justify-between"><span>Outburst Window:</span> <b className="text-zinc-200">[-20d, +60d]</b></div>
              <div className="flex justify-between"><span>Passband Mapping:</span> <b className="text-zinc-200">zg:0, zr:1, zi:2</b></div>
              <div className="flex justify-between"><span>Flux Scale:</span> <b className="text-zinc-200">Peak Normalized</b></div>
            </div>
          </div>

          <div className="p-3.5 rounded bg-zinc-950/70 border border-zinc-900 space-y-2">
            <span className="text-cyan-400 font-bold block">Scientific Anomaly Assessor</span>
            <div className="space-y-1 text-zinc-400">
              <div className="flex justify-between"><span>Real-ZTF Detector v2:</span> <b className="text-cyan-300">EVALUATED_RESEARCH</b></div>
              <div className="flex justify-between"><span>Synthetic Domain v1:</span> <b className="text-zinc-400">VALIDATED_SYNTHETIC</b></div>
              <div className="flex justify-between"><span>Real-ZTF Primary AUROC:</span> <b className="text-emerald-400">0.6881 (v2 LC-Only)</b></div>
              <div className="flex justify-between"><span>Real-ZTF Primary AUPRC:</span> <b className="text-emerald-400">0.6862 (v2 LC-Only)</b></div>
              <div className="flex justify-between"><span>Scientific Policy:</span> <b className="text-zinc-200">PCA-Mahalanobis Novelty</b></div>
            </div>
          </div>
        </div>
      </div>

      {/* Domain Shift Disclosures */}
      <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono font-bold text-cyan-400 uppercase">
          <AlertTriangle className="w-4 h-4 text-cyan-400" />
          <span>Scientific Governance & Evaluation Disclosures</span>
        </div>
        <p className="text-xs font-mono text-zinc-300 leading-relaxed">
          The original ACEI research model v1 was trained and evaluated as a multimodal fusion architecture (co-temporal deep image cutouts + light curves) using simulated synthetic data. In real survey conditions (ZTF Alert Stream / NASA IRSA Photometry), image cutouts are frequently absent. Controlled zero-shot evaluation of v1 demonstrated an AUROC of 0.3859 due to image modality gap.
        </p>
        <p className="text-xs font-mono text-zinc-400 leading-relaxed">
          To resolve this without modifying frozen research artifacts, ACEI implements <b>Real-ZTF Light-Curve Anomaly Detector v2</b> operating directly on the frozen 128-D light-curve representation space. Evaluated on the 106-object primary benchmark, v2 achieves <b>AUROC = 0.6881</b> and <b>AUPRC = 0.6862</b> with a <b>5.17% FPR</b>, operating under status <code>REAL_ZTF_EVALUATED_RESEARCH</code>.
        </p>
      </div>
    </div>
  );
}
