"use client";

import React, { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface FollowupTarget3DProps {
  targetId?: string;
  recommendedModes?: string[];
}

function TargetMesh({ modes }: { modes: string[] }) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const ringsRef = useRef<THREE.Group>(null!);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.3;
      meshRef.current.rotation.x += delta * 0.1;
    }
    if (ringsRef.current) {
      ringsRef.current.rotation.z -= delta * 0.2;
    }
  });

  const hasSpec = modes.some(m => m.toLowerCase().includes("spec"));
  const hasIR = modes.some(m => m.toLowerCase().includes("ir") || m.toLowerCase().includes("infrared"));
  const color = hasSpec ? "#38bdf8" : hasIR ? "#fbbf24" : "#a1a1aa";

  return (
    <group>
      {/* Central Target Sphere */}
      <mesh ref={meshRef}>
        <octahedronGeometry args={[1.2, 2]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3}
          wireframe={true}
        />
      </mesh>

      {/* Target Aiming Concentric Rings */}
      <group ref={ringsRef}>
        <mesh>
          <ringGeometry args={[2.0, 2.05, 32]} />
          <meshBasicMaterial color="#38bdf8" side={THREE.DoubleSide} transparent opacity={0.6} />
        </mesh>
        <mesh rotation={[Math.PI / 3, 0, 0]}>
          <ringGeometry args={[2.5, 2.55, 32]} />
          <meshBasicMaterial color="#94a3b8" side={THREE.DoubleSide} transparent opacity={0.4} />
        </mesh>
      </group>
    </group>
  );
}

export default function FollowupTarget3D({
  targetId = "CAND_SNIa_002",
  recommendedModes = ["Optical Photometry", "Classification Spectroscopy", "Host Galaxy Survey"]
}: FollowupTarget3DProps) {
  return (
    <div className="relative rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-2 overflow-hidden">
      <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2 z-10 relative">
        <div>
          <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider block">
            FOLLOW-UP TARGET MODEL
          </span>
          <span className="text-[10px] font-mono text-zinc-500">
            Conceptual Observation Target Geometry: {targetId}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] font-mono text-zinc-400">
          <span>ACTIVE MODES: {recommendedModes.length}</span>
        </div>
      </div>

      <div className="h-44 w-full relative rounded bg-zinc-950/80 border border-zinc-900 overflow-hidden">
        <Canvas
          dpr={[1, 1.5]}
          camera={{ position: [0, 0, 7], fov: 45 }}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
        >
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 5, 5]} intensity={1} color="#38bdf8" />
          <TargetMesh modes={recommendedModes} />
        </Canvas>

        {/* Mode Indicators */}
        <div className="absolute bottom-2 left-2 right-2 flex flex-wrap gap-1.5 text-[9px] font-mono text-zinc-300 pointer-events-none">
          {recommendedModes.map((mode, idx) => (
            <span
              key={idx}
              className="px-2 py-0.5 rounded bg-zinc-900/90 border border-zinc-800 text-cyan-300"
            >
              {mode}
            </span>
          ))}
        </div>
      </div>

      <p className="text-[10px] font-mono text-zinc-500 italic">
        Conceptual planning visualization — not a telescope scheduling interface.
      </p>
    </div>
  );
}
