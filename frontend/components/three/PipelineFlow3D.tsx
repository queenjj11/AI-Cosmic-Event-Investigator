"use client";

import React, { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

const STAGES = [
  { id: "real_ztf", label: "REAL ZTF", sub: "Photometry", x: -7.5 },
  { id: "qc", label: "QC", sub: "Filtering", x: -5.0 },
  { id: "encoder", label: "128-D ENCODER", sub: "Transformer", x: -2.5 },
  { id: "novelty", label: "NOVELTY", sub: "v2 Detector", x: 0.0 },
  { id: "investigator", label: "INVESTIGATOR", sub: "Characterize", x: 2.5 },
  { id: "literature", label: "LITERATURE", sub: "RAG Retrieval", x: 5.0 },
  { id: "followup", label: "FOLLOW-UP", sub: "Recommender", x: 7.5 },
];

function NodesAndLines() {
  const lineRef = useRef<THREE.LineSegments>(null!);
  const pulsesRef = useRef<THREE.Group>(null!);

  const linePositions = useMemo(() => {
    const pos: number[] = [];
    for (let i = 0; i < STAGES.length - 1; i++) {
      pos.push(STAGES[i].x, 0, 0);
      pos.push(STAGES[i + 1].x, 0, 0);
    }
    return new Float32Array(pos);
  }, []);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (pulsesRef.current) {
      pulsesRef.current.children.forEach((child, idx) => {
        child.position.y = Math.sin(t * 2 + idx * 0.8) * 0.15;
      });
    }
  });

  return (
    <group>
      {/* Connector lines */}
      <lineSegments ref={lineRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[linePositions, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#1e293b" linewidth={1} />
      </lineSegments>

      {/* Stage Nodes */}
      <group ref={pulsesRef}>
        {STAGES.map((stage) => {
          const isCore = stage.id === "encoder" || stage.id === "novelty";
          const color = isCore ? "#38bdf8" : "#94a3b8";
          return (
            <mesh key={stage.id} position={[stage.x, 0, 0]}>
              <sphereGeometry args={[isCore ? 0.35 : 0.25, 16, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={isCore ? 0.5 : 0.2}
                roughness={0.3}
              />
            </mesh>
          );
        })}
      </group>
    </group>
  );
}

export default function PipelineFlow3D() {
  return (
    <div className="relative rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-3 overflow-hidden">
      <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2">
        <div>
          <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider block">
            ACEI INVESTIGATION PIPELINE TOPOLOGY
          </span>
          <span className="text-[10px] font-mono text-zinc-500">
            End-to-End Real-ZTF Light-Curve Processing Architecture
          </span>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800/80 font-bold">
          OPERATIONAL: LC-FIRST v1.0
        </span>
      </div>

      <div className="h-28 w-full relative rounded bg-zinc-950/90 border border-zinc-900 overflow-hidden">
        <Canvas
          dpr={[1, 1.5]}
          camera={{ position: [0, 0, 10], fov: 50 }}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[0, 5, 5]} intensity={1} color="#38bdf8" />
          <NodesAndLines />
        </Canvas>

        {/* Labels Overlay */}
        <div className="absolute inset-x-2 bottom-2 flex justify-between items-center text-[9px] font-mono text-zinc-400 pointer-events-none">
          {STAGES.map((s) => (
            <div key={s.id} className="text-center w-16">
              <div className="font-bold text-zinc-200 truncate">{s.label}</div>
              <div className="text-[8px] text-zinc-500 truncate">{s.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
