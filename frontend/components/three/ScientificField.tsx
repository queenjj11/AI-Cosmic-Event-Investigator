"use client";

import React, { useRef, useMemo, useEffect, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

function StarFieldPoints({ count = 600 }: { count?: number }) {
  const pointsRef = useRef<THREE.Points>(null!);

  // Generate sparse, restrained star field positions & colors
  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const colorChoices = [
      new THREE.Color("#38bdf8"), // Ice cyan
      new THREE.Color("#7dd3fc"), // Soft sky
      new THREE.Color("#e2e8f0"), // Pale white-blue
      new THREE.Color("#94a3b8"), // Muted slate
    ];

    for (let i = 0; i < count; i++) {
      // Distribute in a spherical volume
      const r = 20 + Math.random() * 60;
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos(2 * Math.random() - 1);

      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);

      const c = colorChoices[Math.floor(Math.random() * colorChoices.length)];
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
    }
    return [pos, col];
  }, [count]);

  useFrame((_, delta) => {
    if (pointsRef.current) {
      // Extremely slow, restrained rotation
      pointsRef.current.rotation.y += delta * 0.015;
      pointsRef.current.rotation.x += delta * 0.005;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={1.2}
        sizeAttenuation={true}
        vertexColors={true}
        transparent={true}
        opacity={0.6}
        depthWrite={false}
      />
    </points>
  );
}

function GridPlane() {
  return (
    <gridHelper
      args={[100, 40, "#1e293b", "#0f172a"]}
      position={[0, -20, 0]}
      rotation={[0, 0, 0]}
    />
  );
}

export default function ScientificField() {
  const [reducedMotion, setReducedMotion] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  if (hasError || reducedMotion) {
    // Static CSS fallback for reduced motion or WebGL fallback
    return (
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden bg-[#07090e]">
        <div className="absolute inset-0 opacity-20 scientific-grid" />
        <div className="absolute top-1/4 left-1/3 w-96 h-96 bg-cyan-950/20 rounded-full blur-3xl pointer-events-none" />
      </div>
    );
  }

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden bg-[#07090e]">
      <div className="absolute inset-0 opacity-30 scientific-grid" />
      <Canvas
        dpr={[1, 1.5]}
        camera={{ position: [0, 0, 40], fov: 60 }}
        gl={{ alpha: true, antialias: false, powerPreference: "low-power" }}
        onError={() => setHasError(true)}
      >
        <ambientLight intensity={0.2} />
        <StarFieldPoints count={500} />
        <GridPlane />
      </Canvas>
    </div>
  );
}
