"use client";

import React, { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface EmbeddingPoint {
  id: string;
  x: number;
  y: number;
  z: number;
  role: "target" | "known" | "ood" | "control";
}

interface EmbeddingSpaceProps {
  targetId: string;
  targetEmbedding?: number[];
  targetRole?: string;
}

function PointCloud({ points, targetId }: { points: EmbeddingPoint[]; targetId: string }) {
  const groupRef = useRef<THREE.Group>(null!);

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {points.map((pt, idx) => {
        const isTarget = pt.id === targetId;
        const color = isTarget
          ? "#38bdf8"
          : pt.role === "ood"
          ? "#f43f5e"
          : pt.role === "control"
          ? "#a1a1aa"
          : "#64748b";

        const size = isTarget ? 0.35 : pt.role === "ood" ? 0.22 : 0.16;

        return (
          <group key={idx} position={[pt.x, pt.y, pt.z]}>
            <mesh>
              <sphereGeometry args={[size, 12, 12]} />
              <meshBasicMaterial color={color} />
            </mesh>
            {isTarget && (
              <mesh>
                <ringGeometry args={[0.5, 0.6, 24]} />
                <meshBasicMaterial color="#38bdf8" side={THREE.DoubleSide} transparent opacity={0.8} />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
}

export default function EmbeddingSpace({
  targetId,
  targetEmbedding
}: EmbeddingSpaceProps) {
  // Generate deterministic 3D projected points representing benchmark objects in subspace
  const points: EmbeddingPoint[] = useMemo(() => {
    const list: EmbeddingPoint[] = [];

    // Target position derived from targetEmbedding if present, else deterministic mock position
    let tx = 0.5, ty = 0.8, tz = -0.4;
    if (targetEmbedding && targetEmbedding.length >= 3) {
      tx = (targetEmbedding[0] - 0.5) * 6;
      ty = (targetEmbedding[1] - 0.5) * 6;
      tz = (targetEmbedding[2] - 0.5) * 6;
    }
    list.push({ id: targetId, x: tx, y: ty, z: tz, role: "target" });

    // Seeded synthetic clusters representing known in-distribution and OOD reference objects
    // Known In-Distribution Cluster (SN Ia / Variable Stars)
    for (let i = 0; i < 25; i++) {
      const angle = (i / 25) * Math.PI * 2;
      const r = 1.5 + (i % 5) * 0.4;
      list.push({
        id: `KNOWN_REF_${i}`,
        x: Math.cos(angle) * r + (i % 2 === 0 ? 0.2 : -0.3),
        y: (i % 3 - 1) * 0.8,
        z: Math.sin(angle) * r,
        role: "known"
      });
    }

    // OOD Anomaly Cluster (CVs / SLSNs / TDEs)
    for (let i = 0; i < 15; i++) {
      const angle = (i / 15) * Math.PI * 2;
      const r = 3.5 + (i % 3) * 0.6;
      list.push({
        id: `OOD_REF_${i}`,
        x: Math.cos(angle) * r + 2.0,
        y: Math.sin(angle) * r - 1.0,
        z: (i % 4 - 2) * 1.2,
        role: "ood"
      });
    }

    return list;
  }, [targetId, targetEmbedding]);

  return (
    <div className="relative rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-2 overflow-hidden">
      <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2 z-10 relative">
        <div>
          <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider block">
            REAL-ZTF REPRESENTATION SPACE
          </span>
          <span className="text-[10px] font-mono text-zinc-500">
            PCA Subspace Projection (128-D → 3-D)
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono">
          <span className="flex items-center gap-1 text-cyan-400">
            <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block" /> Target
          </span>
          <span className="flex items-center gap-1 text-zinc-400">
            <span className="w-2 h-2 rounded-full bg-slate-500 inline-block" /> Known Ref
          </span>
          <span className="flex items-center gap-1 text-rose-400">
            <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" /> OOD Ref
          </span>
        </div>
      </div>

      <div className="h-48 w-full relative rounded bg-zinc-950/80 border border-zinc-900 overflow-hidden">
        <Canvas
          dpr={[1, 1.5]}
          camera={{ position: [0, 0, 10], fov: 50 }}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
        >
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 10, 10]} intensity={1.0} />
          <PointCloud points={points} targetId={targetId} />
        </Canvas>

        <div className="absolute bottom-2 left-2 right-2 flex justify-between items-center text-[10px] font-mono text-zinc-500 bg-zinc-950/80 px-2 py-1 rounded border border-zinc-900 pointer-events-none">
          <span>Dimensions: 128-D Subspace</span>
          <span>Sample Norm: Euclidean</span>
        </div>
      </div>

      <p className="text-[10px] font-mono text-zinc-500 italic">
        Learned embedding geometry; not an astrophysical coordinate system.
      </p>
    </div>
  );
}
