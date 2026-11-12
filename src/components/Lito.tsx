"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import type { FaceWeights } from "@/lib/a2f";

const MODEL_URL = "/models/lito.glb";

type LitoProps = {
  /** true while the man should be animating speech */
  talking: boolean;
  /** incremented on each spoken word boundary to emphasise the mouth */
  wordPulse: number;
  /** Audio2Face-3D blendshape weights for this frame, when audio-driven playback is active. */
  face?: () => FaceWeights | null;
};

// The scan faces this many radians away from +Z once baked.
const YAW = 0.77;

// Facial landmarks in the baked/normalised model space (found by profiling
// the face midline: nose tip y≈0.585, lips meet y≈0.512, chin y≈0.44).
const CENTER_X = 0.012;
const SLIT_Y = 0.512;
const JAW_MAX = 0.09;

/**
 * The model is a photogrammetry scan with no rig, so speech is faked in the
 * vertex shader: everything below the lip line rotates about a jaw hinge, the
 * gap between the lips is darkened, and the head pivots on the neck.
 */
function makeUniforms() {
  return {
    uJaw: { value: 0 },
    uHeadRot: { value: new THREE.Matrix3() },
  };
}

const VERTEX_PARS = /* glsl */ `
uniform float uJaw;
uniform mat3 uHeadRot;
varying vec3 vRest;
`;

const VERTEX_MAIN = /* glsl */ `
vec3 rest = transformed;
vRest = rest;

// Jaw: below the lip line, fading out down the neck, toward the back of the
// head and past the sides of the jaw.
float below = 1.0 - smoothstep(${SLIT_Y - 0.008}, ${SLIT_Y + 0.002}, rest.y);
float chin = smoothstep(0.32, 0.40, rest.y);
float front = smoothstep(0.02, 0.12, rest.z);
float side = 1.0 - smoothstep(0.13, 0.19, abs(rest.x - ${CENTER_X}));
float a = uJaw * below * chin * front * side;
vec3 jawPivot = vec3(${CENTER_X}, 0.57, 0.08);
vec3 d = transformed - jawPivot;
transformed = jawPivot + vec3(d.x, d.y * cos(a) - d.z * sin(a), d.y * sin(a) + d.z * cos(a));

// Head turns on the neck, leaving the shoulders still.
float hw = smoothstep(0.28, 0.42, rest.y);
vec3 headPivot = vec3(${CENTER_X}, 0.36, 0.05);
transformed = mix(transformed, headPivot + uHeadRot * (transformed - headPivot), hw);
`;

const FRAGMENT_PARS = /* glsl */ `
uniform float uJaw;
varying vec3 vRest;
`;

const FRAGMENT_MAIN = /* glsl */ `
// Darken the stretched lip seam so the open mouth reads as a cavity.
float seam = 1.0 - smoothstep(0.002, 0.0045, abs(vRest.y - ${SLIT_Y - 0.003}));
float mouthW = 1.0 - smoothstep(0.03, 0.055, abs(vRest.x - ${CENTER_X}));
float mouthFront = smoothstep(0.26, 0.30, vRest.z);
float cavity = seam * mouthW * mouthFront * clamp(uJaw / ${JAW_MAX}, 0.0, 1.0);
diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.1, 0.02, 0.02), cavity * 0.95);
`;

function buildModel(source: THREE.Object3D, uniforms: ReturnType<typeof makeUniforms>) {
  source.updateMatrixWorld(true);
  const group = new THREE.Group();
  const materials = new Map<THREE.Material, THREE.Material>();

  source.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const original = mesh.material as THREE.Material;
    if (!materials.has(original)) {
      const material = original.clone();
      material.onBeforeCompile = (shader) => {
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = shader.vertexShader
          .replace("#include <common>", `#include <common>\n${VERTEX_PARS}`)
          .replace("#include <begin_vertex>", `#include <begin_vertex>\n${VERTEX_MAIN}`);
        shader.fragmentShader = shader.fragmentShader
          .replace("#include <common>", `#include <common>\n${FRAGMENT_PARS}`)
          .replace("#include <map_fragment>", `#include <map_fragment>\n${FRAGMENT_MAIN}`);
      };
      material.customProgramCacheKey = () => "lito-talking";
      materials.set(original, material);
    }
    const geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
    const baked = new THREE.Mesh(geometry, materials.get(original));
    baked.frustumCulled = false;
    group.add(baked);
  });

  // Normalise: centred on X/Z, standing on y = 0, facing +Z.
  const box = new THREE.Box3().setFromObject(group);
  const c = box.getCenter(new THREE.Vector3());
  group.children.forEach((m) =>
    (m as THREE.Mesh).geometry.translate(-c.x, -box.min.y, -c.z).rotateY(YAW)
  );
  return group;
}

export default function Lito({ talking, wordPulse, face }: LitoProps) {
  const { scene } = useGLTF(MODEL_URL);
  const uniforms = useMemo(() => makeUniforms(), []);
  const model = useMemo(() => buildModel(scene, uniforms), [scene, uniforms]);
  // useFrame mutates the uniforms every frame, so read them through a ref.
  const uniformsRef = useRef<ReturnType<typeof makeUniforms>>(null);
  useEffect(() => {
    uniformsRef.current = uniforms;
  }, [uniforms]);

  const state = useRef({
    open: 0,
    talk: 0,
    lastPulse: 0,
    pulseTime: -10,
    euler: new THREE.Euler(),
    matrix: new THREE.Matrix4(),
  });

  useFrame(({ clock }, delta) => {
    const u = uniformsRef.current;
    if (!u) return;
    const t = clock.elapsedTime;
    const s = state.current;

    if (wordPulse !== s.lastPulse) {
      s.lastPulse = wordPulse;
      s.pulseTime = t;
    }

    // Overall talking intensity eases in/out.
    s.talk = THREE.MathUtils.damp(s.talk, talking ? 1 : 0, 4, delta);

    // Syllable-like mouth motion: fast oscillation, modulated by a slower
    // envelope, with short pauses between "phrases".
    const syllable = Math.max(0, Math.sin(t * 12.5) * 0.6 + Math.sin(t * 7.3) * 0.4);
    const envelope = 0.55 + 0.45 * Math.sin(t * 2.1 + Math.sin(t * 0.9) * 2);
    const phrase = THREE.MathUtils.smoothstep(Math.sin(t * 0.7) + Math.sin(t * 1.9) * 0.4, -0.9, -0.4);
    const pulse = Math.max(0, 1 - (t - s.pulseTime) * 6);
    const target = s.talk * Math.min(1, syllable * envelope * phrase + pulse * 0.5);
    s.open = THREE.MathUtils.damp(s.open, target, 18, delta);
    // Only the jaw can move on this scan, so Audio2Face drives just jawOpen
    // (minus mouthClose, which presses the lips together over an open jaw).
    const w = face?.();
    const open = w ? Math.max(0, (w.jawopen ?? 0) - (w.mouthclose ?? 0) * 0.5) : s.open;
    u.uJaw.value = Math.min(1, open) * JAW_MAX;

    // Head motion: gentle idle sway plus nods while talking.
    s.euler.set(
      Math.sin(t * 1.7) * 0.04 * s.talk + Math.sin(t * 0.4) * 0.012,
      Math.sin(t * 0.5) * 0.1 + Math.sin(t * 1.3) * 0.03 * s.talk,
      Math.sin(t * 0.8) * 0.025,
      "YXZ"
    );
    u.uHeadRot.value.setFromMatrix4(s.matrix.makeRotationFromEuler(s.euler));
  });

  return <primitive object={model} />;
}
