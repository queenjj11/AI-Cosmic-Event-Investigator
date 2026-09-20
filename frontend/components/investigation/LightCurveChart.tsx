"use client";

import React, { useState, useMemo } from "react";
import { ZoomIn, ZoomOut, RotateCcw, Filter } from "lucide-react";
import { LightCurvePoint } from "@/types/investigator";

interface LightCurveChartProps {
  points: LightCurvePoint[];
  candidateId?: string;
}

const BAND_COLORS: Record<string, { stroke: string; fill: string; label: string }> = {
  g: { stroke: "#10b981", fill: "rgba(16, 185, 129, 0.25)", label: "ZTF g (zg)" },
  zg: { stroke: "#10b981", fill: "rgba(16, 185, 129, 0.25)", label: "ZTF g (zg)" },
  r: { stroke: "#f43f5e", fill: "rgba(244, 63, 94, 0.25)", label: "ZTF r (zr)" },
  zr: { stroke: "#f43f5e", fill: "rgba(244, 63, 94, 0.25)", label: "ZTF r (zr)" },
  i: { stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.25)", label: "ZTF i (zi)" },
  zi: { stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.25)", label: "ZTF i (zi)" },
};

export function LightCurveChart({ points, candidateId }: LightCurveChartProps) {
  const [activeBands, setActiveBands] = useState<Record<string, boolean>>({
    g: true,
    zg: true,
    r: true,
    zr: true,
    i: true,
    zi: true,
  });

  const [hoveredPoint, setHoveredPoint] = useState<LightCurvePoint | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1); // 1x, 1.5x, 2x, 3x

  // Filter visible points
  const visiblePoints = useMemo(() => {
    return points.filter((p) => activeBands[p.band] !== false);
  }, [points, activeBands]);

  // Dimension and scale calculations
  const width = 840;
  const height = 400;
  const padding = { top: 35, right: 35, bottom: 50, left: 65 };

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  // Compute peak point for scientific annotation
  const peakPoint = useMemo(() => {
    if (visiblePoints.length === 0) return null;
    return visiblePoints.reduce((max, p) => (p.norm_flux > max.norm_flux ? p : max), visiblePoints[0]);
  }, [visiblePoints]);

  const { minT, maxT, minF, maxF } = useMemo(() => {
    if (points.length === 0) {
      return { minT: 0, maxT: 80, minF: 0, maxF: 1.2 };
    }
    const times = points.map((p) => p.rel_time);
    const fluxes = points.map((p) => p.norm_flux);
    const min_t = Math.min(...times);
    const max_t = Math.max(...times);
    const min_f = Math.min(...fluxes);
    const max_f = Math.max(...fluxes);
    const t_span = (max_t - min_t || 1) / zoomLevel;
    const f_span = (max_f - min_f || 1) / zoomLevel;

    const mid_t = (min_t + max_t) / 2;
    const mid_f = (min_f + max_f) / 2;

    return {
      minT: Math.max(0, mid_t - t_span * 0.55),
      maxT: mid_t + t_span * 0.55,
      minF: Math.max(0, mid_f - f_span * 0.55),
      maxF: mid_f + f_span * 0.60,
    };
  }, [points, zoomLevel]);

  const scaleX = (t: number) => padding.left + ((t - minT) / (maxT - minT || 1)) * plotWidth;
  const scaleY = (f: number) => padding.top + plotHeight - ((f - minF) / (maxF - minF || 1)) * plotHeight;

  // Grid tick lines
  const xTicks = useMemo(() => {
    const ticks = [];
    const step = (maxT - minT) / 6;
    for (let i = 0; i <= 6; i++) {
      ticks.push(minT + i * step);
    }
    return ticks;
  }, [minT, maxT]);

  const yTicks = useMemo(() => {
    const ticks = [];
    const step = (maxF - minF) / 5;
    for (let i = 0; i <= 5; i++) {
      ticks.push(minF + i * step);
    }
    return ticks;
  }, [minF, maxF]);

  const toggleBand = (b: string) => {
    setActiveBands((prev) => ({ ...prev, [b]: !prev[b] }));
  };

  const uniqueBands = useMemo(() => {
    const s = new Set<string>();
    points.forEach((p) => s.add(p.band));
    return Array.from(s);
  }, [points]);

  const handleZoom = (delta: number) => {
    setZoomLevel((prev) => Math.min(4, Math.max(1, +(prev + delta).toFixed(1))));
  };

  const resetZoom = () => setZoomLevel(1);

  return (
    <div className="rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-4">
      {/* Header controls & filter toggles */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs font-mono text-zinc-400">
            <Filter className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-bold text-white uppercase">Passbands:</span>
          </div>
          <div className="flex gap-2">
            {uniqueBands.map((band) => {
              const conf = BAND_COLORS[band] || { stroke: "#38bdf8", fill: "rgba(56, 189, 248, 0.2)", label: band };
              const isActive = activeBands[band] !== false;
              return (
                <button
                  key={band}
                  onClick={() => toggleBand(band)}
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono transition-all border ${
                    isActive
                      ? "bg-zinc-900 border-zinc-700 text-white shadow-sm"
                      : "bg-zinc-950 border-zinc-800 text-zinc-600 line-through"
                  }`}
                >
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: conf.stroke }} />
                  <span>{conf.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Zoom Toolbar */}
          <div className="flex items-center gap-1 p-1 rounded bg-zinc-900/90 border border-zinc-800 text-xs font-mono">
            <button
              onClick={() => handleZoom(0.5)}
              disabled={zoomLevel >= 4}
              className="p-1 text-zinc-400 hover:text-cyan-300 disabled:opacity-30 transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <span className="px-1.5 text-[11px] text-cyan-400 font-bold">{zoomLevel.toFixed(1)}x</span>
            <button
              onClick={() => handleZoom(-0.5)}
              disabled={zoomLevel <= 1}
              className="p-1 text-zinc-400 hover:text-cyan-300 disabled:opacity-30 transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            {zoomLevel > 1 && (
              <button
                onClick={resetZoom}
                className="p-1 text-zinc-400 hover:text-cyan-300 transition-colors ml-1 border-l border-zinc-800 pl-1.5"
                title="Reset Zoom"
              >
                <RotateCcw className="w-3 h-3" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-2 text-xs font-mono text-zinc-400">
            {candidateId && <span className="text-zinc-500 font-semibold">{candidateId}</span>}
            <span>Showing <span className="text-cyan-400 font-bold">{visiblePoints.length}</span> / {points.length} Photometric Epochs</span>
          </div>
        </div>
      </div>

      {/* SVG Scientific Canvas */}
      <div className="relative w-full overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-h-[440px] select-none">
          {/* Background Grid */}
          <rect x={padding.left} y={padding.top} width={plotWidth} height={plotHeight} fill="#060910" stroke="#1f293d" strokeWidth="1" />

          {/* Horizontal Grid lines */}
          {yTicks.map((yVal, idx) => {
            const y = scaleY(yVal);
            return (
              <g key={`y-${idx}`}>
                <line x1={padding.left} y1={y} x2={padding.left + plotWidth} y2={y} stroke="#131b2e" strokeWidth="1" strokeDasharray="3 3" />
                <text x={padding.left - 8} y={y + 3} textAnchor="end" fill="#64748b" fontSize="10" fontFamily="monospace">
                  {yVal.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Vertical Grid lines */}
          {xTicks.map((xVal, idx) => {
            const x = scaleX(xVal);
            return (
              <g key={`x-${idx}`}>
                <line x1={x} y1={padding.top} x2={x} y2={padding.top + plotHeight} stroke="#131b2e" strokeWidth="1" strokeDasharray="3 3" />
                <text x={x} y={padding.top + plotHeight + 18} textAnchor="middle" fill="#64748b" fontSize="10" fontFamily="monospace">
                  {xVal.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Axes Labels */}
          <text
            x={padding.left + plotWidth / 2}
            y={height - 10}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="11"
            fontFamily="monospace"
            fontWeight="bold"
          >
            Relative Time (Days from First Observation Epoch)
          </text>
          <text
            x={18}
            y={padding.top + plotHeight / 2}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="11"
            fontFamily="monospace"
            fontWeight="bold"
            transform={`rotate(-90 18 ${padding.top + plotHeight / 2})`}
          >
            Normalized Flux
          </text>

          {/* Error Bars and Data Points */}
          {visiblePoints.map((pt, idx) => {
            const cx = scaleX(pt.rel_time);
            const cy = scaleY(pt.norm_flux);
            const errHeight = ((pt.norm_flux_err || 0.01) / (maxF - minF || 1)) * plotHeight;
            const yErrTop = cy - errHeight;
            const yErrBottom = cy + errHeight;

            const conf = BAND_COLORS[pt.band] || { stroke: "#38bdf8", fill: "rgba(56, 189, 248, 0.2)", label: pt.band };
            const isHovered = hoveredPoint === pt;

            return (
              <g
                key={idx}
                onMouseEnter={() => setHoveredPoint(pt)}
                onMouseLeave={() => setHoveredPoint(null)}
                className="cursor-pointer"
              >
                {/* Vertical Error Bar */}
                <line x1={cx} y1={yErrTop} x2={cx} y2={yErrBottom} stroke={conf.stroke} strokeWidth="1" strokeOpacity="0.75" />
                {/* Error Bar Caps */}
                <line x1={cx - 2.5} y1={yErrTop} x2={cx + 2.5} y2={yErrTop} stroke={conf.stroke} strokeWidth="1" strokeOpacity="0.75" />
                <line x1={cx - 2.5} y1={yErrBottom} x2={cx + 2.5} y2={yErrBottom} stroke={conf.stroke} strokeWidth="1" strokeOpacity="0.75" />

                {/* Observation Marker Circle */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 6 : 3.5}
                  fill={conf.stroke}
                  stroke="#0a0e17"
                  strokeWidth="1.5"
                  className="transition-all duration-150"
                />
              </g>
            );
          })}

          {/* Scientific Annotation: Peak Flux Marker */}
          {peakPoint && (
            <g>
              <circle
                cx={scaleX(peakPoint.rel_time)}
                cy={scaleY(peakPoint.norm_flux)}
                r="9"
                fill="none"
                stroke="#38bdf8"
                strokeWidth="1.5"
                strokeDasharray="2 2"
                className="animate-pulse"
              />
              <line
                x1={scaleX(peakPoint.rel_time)}
                y1={scaleY(peakPoint.norm_flux) - 10}
                x2={scaleX(peakPoint.rel_time)}
                y2={scaleY(peakPoint.norm_flux) - 24}
                stroke="#38bdf8"
                strokeWidth="1"
              />
              <text
                x={scaleX(peakPoint.rel_time)}
                y={scaleY(peakPoint.norm_flux) - 28}
                textAnchor="middle"
                fill="#38bdf8"
                fontSize="9"
                fontFamily="monospace"
                fontWeight="bold"
              >
                PEAK FLUX ({peakPoint.norm_flux.toFixed(3)})
              </text>
            </g>
          )}

          {/* Hover Crosshairs */}
          {hoveredPoint && (
            <g>
              <line
                x1={padding.left}
                y1={scaleY(hoveredPoint.norm_flux)}
                x2={padding.left + plotWidth}
                y2={scaleY(hoveredPoint.norm_flux)}
                stroke="#38bdf8"
                strokeWidth="1"
                strokeDasharray="4 4"
                strokeOpacity="0.6"
              />
              <line
                x1={scaleX(hoveredPoint.rel_time)}
                y1={padding.top}
                x2={scaleX(hoveredPoint.rel_time)}
                y2={padding.top + plotHeight}
                stroke="#38bdf8"
                strokeWidth="1"
                strokeDasharray="4 4"
                strokeOpacity="0.6"
              />
            </g>
          )}
        </svg>

        {/* Hover Tooltip Overlay */}
        {hoveredPoint && (
          <div
            className="absolute z-10 p-2.5 rounded bg-[#090d16]/95 border border-cyan-500/80 shadow-xl text-[11px] font-mono pointer-events-none space-y-1"
            style={{
              left: Math.min(plotWidth - 100, Math.max(10, scaleX(hoveredPoint.rel_time) - 75)),
              top: Math.max(10, scaleY(hoveredPoint.norm_flux) - 95),
            }}
          >
            <div className="flex items-center justify-between gap-3 text-cyan-400 font-bold border-b border-zinc-800 pb-1">
              <span>Band: {hoveredPoint.band.toUpperCase()}</span>
              <span>t = {hoveredPoint.rel_time.toFixed(2)} d</span>
            </div>
            <div className="text-zinc-200">
              Flux: <span className="font-bold">{hoveredPoint.norm_flux.toFixed(4)}</span>
            </div>
            <div className="text-zinc-400">
              Uncertainty: ± {hoveredPoint.norm_flux_err ? hoveredPoint.norm_flux_err.toFixed(4) : "N/A"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
