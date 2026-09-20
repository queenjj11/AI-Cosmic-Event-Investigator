"use client";

import React, { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface EventCoreProps {
  candidateId: string;
  noveltyScore?: number | null;
  isAnomaly?: boolean | null;
  tokenCount?: number;
  embeddingNorm?: number;
}

function CoreGeometry({ noveltyScore, isAnomaly }: { noveltyScore?: number | null; isAnomaly?: boolean | null }) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const wireframeRef = useRef<THREE.Mesh>(null!);

  const color = useMemo(() => {
    if (isAnomaly) return new THREE.Color("#f43f5e"); // Rose for anomaly
    if (noveltyScore !== undefined && noveltyScore !== null) {
      if (noveltyScore > 0.7) return new THREE.Color("#fbbf24"); // Amber
      return new THREE.Color("#38bdf8"); // Cyan
    }
    return new THREE.Color("#38bdf8");
  }, [noveltyScore, isAnomaly]);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.4;
      meshRef.current.rotation.x += delta * 0.2;
    }
    if (wireframeRef.current) {
      wireframeRef.current.rotation.y -= delta * 0.2;
      wireframeRef.current.rotation.z += delta * 0.1;
    }
  });

  return (
    <group>
      {/* Solid Inner Core */}
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[1.4, 2]} />
        <meshStandardMaterial
          color={color}
          roughness={0.4}
          metalness={0.8}
          wireframe={false}
          emissive={color}
          emissiveIntensity={isAnomaly ? 0.6 : 0.25}
        />
      </mesh>

      {/* Outer Wireframe Hull */}
      <mesh ref={wireframeRef}>
        <icosahedronGeometry args={[1.9, 1]} />
        <meshBasicMaterial
          color={color}
          wireframe={true}
          transparent={true}
          opacity={0.35}
        />
      </mesh>
    </group>
  );
}

function OrbitalRings({ tokenCount = 30 }: { tokenCount?: number }) {
  const ringsRef = useRef<THREE.Group>(null!);

  const particleCount = useMemo(() => Math.min(120, Math.max(30, tokenCount * 2)), [tokenCount]);

  const [ringPositions] = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      const angle = (i / particleCount) * Math.PI * 2;
      const radius = 2.6 + (i % 3) * 0.3;
      pos[i * 3] = Math.cos(angle) * radius;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 0.4;
      pos[i * 3 + 2] = Math.sin(angle) * radius;
    }
    return [pos];
  }, [particleCount]);

  useFrame((_, delta) => {
    if (ringsRef.current) {
      ringsRef.current.rotation.y += delta * 0.3;
      ringsRef.current.rotation.x = Math.sin(Date.now() * 0.001) * 0.15;
    }
  });

  return (
    <group ref={ringsRef}>
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[ringPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.08}
          color="#7dd3fc"
          transparent={true}
          opacity={0.7}
        />
      </points>
    </group>
  );
}

export default function EventCore({
  candidateId,
  noveltyScore,
  isAnomaly,
  tokenCount = 30,
  embeddingNorm
}: EventCoreProps) {
  return (
    <div className="relative rounded-lg bg-[#0a0e17] border border-zinc-800 p-4 space-y-2 overflow-hidden">
      <div className="flex items-center justify-between border-b border-zinc-800/80 pb-2 z-10 relative">
        <div>
          <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider block">
            LEARNED REPRESENTATION
          </span>
          <span className="text-[10px] font-mono text-zinc-500">
            Object: {candidateId}
          </span>
        </div>
        <div className="flex items-center gap-2 text-right">
          {embeddingNorm !== undefined && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">
              ||z|| = {embeddingNorm.toFixed(2)}
            </span>
          )}
        </div>
      </div>

      {/* 3D Canvas Container */}
      <div className="h-48 w-full relative rounded bg-zinc-950/80 border border-zinc-900 overflow-hidden">
        <Canvas
          dpr={[1, 1.5]}
          camera={{ position: [0, 0, 7], fov: 45 }}
          gl={{ alpha: true, antialias: true, powerPreference: "low-power" }}
        >
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 5, 5]} intensity={1.2} />
          <pointLight position={[-5, -5, -5]} intensity={0.5} color="#38bdf8" />
          <CoreGeometry noveltyScore={noveltyScore} isAnomaly={isAnomaly} />
          <OrbitalRings tokenCount={tokenCount} />
        </Canvas>

        {/* Legend Overlay */}
        <div className="absolute bottom-2 left-2 right-2 flex justify-between items-center text-[10px] font-mono text-zinc-500 bg-zinc-950/80 px-2 py-1 rounded border border-zinc-900 pointer-events-none">
          <span>Tokens: {tokenCount}/50</span>
          <span>
            {noveltyScore !== undefined && noveltyScore !== null
              ? `Score: ${noveltyScore.toFixed(4)}`
              : "Score: Uncalibrated"}
          </span>
        </div>
      </div>

      <p className="text-[10px] font-mono text-zinc-500 italic">
        Visualization of model-derived representation; not a physical model of the source.
      </p>
    </div>
  );
}
