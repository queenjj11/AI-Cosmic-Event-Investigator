"use client";

import React, { useState } from "react";
import { RepresentationSummary } from "@/types/investigator";
import { ChevronDown, ChevronUp, Cpu, Info } from "lucide-react";

interface EmbeddingHeatmapProps {
  representation: RepresentationSummary;
}

export function EmbeddingHeatmap({ representation }: EmbeddingHeatmapProps) {
  const [showRaw, setShowRaw] = useState<boolean>(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const embedding = representation.embedding || [];
  const minVal = Math.min(...embedding, -0.1);
  const maxVal = Math.max(...embedding, 0.1);

  // Map float value to diverging color (blue -> cyan -> white/magenta)
  const getColor = (val: number) => {
    if (val < 0) {
      const intensity = Math.min(1, Math.abs(val) / (Math.abs(minVal) || 1));
      return `rgba(59, 130, 246, ${0.2 + intensity * 0.7})`;
    } else {
      const intensity = Math.min(1, val / (maxVal || 1));
      return `rgba(6, 182, 212, ${0.2 + intensity * 0.7})`;
    }
  };

  return (
    <div className="rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-4">
      {/* Header and Telemetry */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-mono font-bold text-white uppercase">
            128-D Latent Sequence Embedding
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div>
            <span className="text-zinc-500">DIM:</span>{" "}
            <span className="text-cyan-400 font-bold">{representation.embedding_dimension}</span>
          </div>
          <div>
            <span className="text-zinc-500">L2 NORM:</span>{" "}
            <span className="text-white font-bold">{representation.embedding_norm.toFixed(4)}</span>
          </div>
          <div>
            <span className="text-zinc-500">TOKENS:</span>{" "}
            <span className="text-emerald-400 font-bold">{representation.valid_token_count}</span>
            <span className="text-zinc-600"> / 50</span>
          </div>
          <div>
            <span className="text-zinc-500">PADDING:</span>{" "}
            <span className="text-zinc-300">{(representation.padding_fraction * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* Scientific Distinction Warning Banner */}
      <div className="flex items-start gap-2 p-2.5 rounded bg-zinc-900/60 border border-zinc-800 text-[11px] font-mono text-zinc-400">
        <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <span>
          <b className="text-zinc-200">Learned representation — not an astrophysical measurement.</b> Extracted via
          the frozen production <code>LightCurveEncoder</code> (Time2Vec continuous time + 3-layer Transformer + masked pooling).
        </span>
      </div>

      {/* 128-D Heatmap Grid (8 rows x 16 cols) */}
      <div className="space-y-1">
        <div className="flex justify-between items-center text-[10px] font-mono text-zinc-500 px-1">
          <span>LATENT CELL (0..127)</span>
          {hoveredIndex !== null && (
            <span className="text-cyan-300 font-bold">
              Index [{hoveredIndex}]: {embedding[hoveredIndex]?.toFixed(6)}
            </span>
          )}
        </div>

        <div className="grid grid-cols-16 sm:grid-cols-32 gap-1 p-2 bg-[#060910] border border-zinc-800/80 rounded">
          {embedding.map((val, idx) => (
            <div
              key={idx}
              className="h-4 rounded-[2px] transition-all hover:scale-125 cursor-pointer relative group border border-transparent hover:border-cyan-300"
              style={{ backgroundColor: getColor(val) }}
              onMouseEnter={() => setHoveredIndex(idx)}
              onMouseLeave={() => setHoveredIndex(null)}
            />
          ))}
        </div>
      </div>

      {/* Expandable Raw Float Array Toggle */}
      <div>
        <button
          onClick={() => setShowRaw(!showRaw)}
          className="flex items-center gap-1.5 text-xs font-mono text-zinc-400 hover:text-cyan-300 transition-colors"
        >
          {showRaw ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          <span>{showRaw ? "Collapse Raw Vector" : "Expand 128-D Raw Vector Array"}</span>
        </button>

        {showRaw && (
          <div className="mt-2 p-3 bg-zinc-950 border border-zinc-900 rounded font-mono text-[11px] text-zinc-400 overflow-x-auto max-h-48">
            <pre>[{embedding.map((v) => v.toFixed(6)).join(", ")}]</pre>
          </div>
        )}
      </div>
    </div>
  );
}
