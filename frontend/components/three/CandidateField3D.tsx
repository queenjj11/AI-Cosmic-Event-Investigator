"use client";

import React, { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useRouter } from "next/navigation";

interface CandidatePoint {
  candidate_id: string;
  ztf_designation?: string;
  class?: string;
  claimed_type?: string;
  ra?: number;
  dec?: number;
  novelty_score?: number | null;
  x: number;
  y: number;
  z: number;
}

interface CandidateField3DProps {
  candidates: any[];
}

function CandidateCloud({
  points,
  onHover,
  onClick
}: {
  points: CandidatePoint[];
  onHover: (pt: CandidatePoint | null) => void;
  onClick: (id: string) => void;
}) {
  const groupRef = useRef<THREE.Group>(null!);

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.05;
    }
  });

  return (
    <group ref={groupRef}>
      {points.map((pt, idx) => {
        const isCV = pt.class?.includes("Cataclysmic") || pt.claimed_type?.includes("CV");
        const isSNIa = pt.class?.includes("SN Ia") || pt.claimed_type?.includes("Ia");

        const color = isCV ? "#f43f5e" : isSNIa ? "#38bdf8" : "#94a3b8";

        return (
          <mesh
            key={idx}
            position={[pt.x, pt.y, pt.z]}
            onPointerOver={(e) => {
              e.stopPropagation();
              onHover(pt);
            }}
            onPointerOut={() => onHover(null)}
            onClick={(e) => {
              e.stopPropagation();
              onClick(pt.candidate_id);
            }}
          >
            <sphereGeometry args={[0.25, 12, 12]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={0.4}
              roughness={0.3}
            />
          </mesh>
        );
      })}
    </group>
  );
}

export default function CandidateField3D({ candidates }: CandidateField3DProps) {
  const router = useRouter();
  const [hovered, setHovered] = useState<CandidatePoint | null>(null);

  const points: CandidatePoint[] = useMemo(() => {
    return candidates.map((c, i) => {
      // Convert RA/Dec or index into spatial coordinates
      const ra = Number(c.ra) || 0;
      const dec = Number(c.dec) || 0;
      const radRa = (ra * Math.PI) / 180;
      const radDec = (dec * Math.PI) / 180;
      const dist = 6 + (i % 5) * 0.5;

      const x = dist * Math.cos(radDec) * Math.sin(radRa);
      const y = dist * Math.sin(radDec);
      const z = dist * Math.cos(radDec) * Math.cos(radRa);

      return {
        candidate_id: c.candidate_id,
        ztf_designation: c.ztf_designation || c.ztf_object_id,
        class: c.class || c.claimed_type,
        claimed_type: c.claimed_type || c.class,
        ra,
        dec,
        novelty_score: c.novelty_score ?? c.raw_novelty_score ?? null,
        x: isNaN(x) ? (i % 7 - 3) * 1.5 : x,
        y: isNaN(y) ? (Math.floor(i / 7) - 3) * 1.5 : y,
        z: isNaN(z) ? 0 : z
      };
    });
  }, [candidates]);

  return (
    <div className="relative rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-2 overflow-hidden">
      <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2 z-10 relative">
        <div>
          <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider block">
            REAL-ZTF SPATIAL CANDIDATE FIELD
          </span>
          <span className="text-[10px] font-mono text-zinc-500">
            Showing {candidates.length} Benchmark Candidates in Projected Sky-Vector Space
          </span>
        </div>
        <div className="text-[10px] font-mono text-zinc-400">
          Click any candidate point to launch workstation investigation
        </div>
      </div>

      <div className="h-[400px] w-full relative rounded bg-zinc-950/90 border border-zinc-900 overflow-hidden cursor-crosshair">
        <Canvas
          dpr={[1, 1.5]}
          camera={{ position: [0, 0, 16], fov: 50 }}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
        >
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 10, 10]} intensity={1} color="#38bdf8" />
          <CandidateCloud
            points={points}
            onHover={setHovered}
            onClick={(id) => router.push(`/investigate/${id}`)}
          />
        </Canvas>

        {/* Hover Information Tooltip */}
        {hovered && (
          <div className="absolute top-4 left-4 p-3 rounded bg-zinc-950/95 border border-cyan-500/80 shadow-lg text-xs font-mono space-y-1 pointer-events-none z-20 min-w-[200px]">
            <div className="text-cyan-400 font-bold">{hovered.candidate_id}</div>
            <div className="text-zinc-300">Designation: {hovered.ztf_designation || "ZTF Candidate"}</div>
            <div className="text-zinc-400">Class: <span className="text-white font-bold">{hovered.class || "Unknown"}</span></div>
            <div className="text-zinc-400">
              RA / DEC: {hovered.ra?.toFixed(3)}°, {hovered.dec?.toFixed(3)}°
            </div>
            {hovered.novelty_score !== null && hovered.novelty_score !== undefined && (
              <div className="text-amber-400 pt-1 border-t border-zinc-800">
                Novelty Score: {hovered.novelty_score.toFixed(4)}
              </div>
            )}
          </div>
        )}
      </div>

      <p className="text-[10px] font-mono text-zinc-500 italic">
        3D spatial visualization of candidate catalog vectors; click point to open full workstation analysis.
      </p>
    </div>
  );
}
