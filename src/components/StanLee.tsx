"use client";

import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { clone } from "three/examples/jsm/utils/SkeletonUtils.js";
import * as THREE from "three";
import type { FaceWeights } from "@/lib/a2f";

const MODEL_URL = "/models/stan_lee.glb";

type StanLeeProps = {
  /** true while the man should be animating speech */
  talking: boolean;
  /** incremented on each spoken word boundary to emphasise the mouth */
  wordPulse: number;
  /** Audio2Face-3D blendshape weights for this frame, when audio-driven playback is active. */
  face?: () => FaceWeights | null;
};

const X = new THREE.Vector3(1, 0, 0);
const Y = new THREE.Vector3(0, 1, 0);
const Z = new THREE.Vector3(0, 0, 1);

/**
 * A bone plus its bind pose, so we can apply rotations/offsets expressed in
 * world-aligned axes regardless of how each bone's local axes are oriented.
 */
type Joint = {
  bone: THREE.Object3D;
  bindQuat: THREE.Quaternion;
  bindPos: THREE.Vector3;
  parentQuat: THREE.Quaternion;
  parentQuatInv: THREE.Quaternion;
  parentScale: number;
};

function makeJoint(bone: THREE.Object3D): Joint {
  const parentQuat = new THREE.Quaternion();
  const parentScale = new THREE.Vector3();
  bone.parent!.matrixWorld.decompose(new THREE.Vector3(), parentQuat, parentScale);
  return {
    bone,
    bindQuat: bone.quaternion.clone(),
    bindPos: bone.position.clone(),
    parentQuat,
    parentQuatInv: parentQuat.clone().invert(),
    parentScale: parentScale.x,
  };
}

const _q = new THREE.Quaternion();
const _qa = new THREE.Quaternion();
const _v = new THREE.Vector3();

/** Rotates a joint from its bind pose by `x`, `y`, `z` radians about world axes. */
function rotate(j: Joint, x: number, y = 0, z = 0) {
  _q.setFromAxisAngle(Y, y).multiply(_qa.setFromAxisAngle(X, x)).multiply(_qa.setFromAxisAngle(Z, z));
  // Convert the world-space rotation into the parent's space.
  _q.premultiply(j.parentQuatInv).multiply(j.parentQuat);
  j.bone.quaternion.copy(_q).multiply(j.bindQuat);
}

/** Moves a joint from its bind position by a world-space offset. */
function offset(j: Joint, x: number, y: number, z: number) {
  _v.set(x, y, z).applyQuaternion(j.parentQuatInv).divideScalar(j.parentScale);
  j.bone.position.copy(j.bindPos).add(_v);
}

function buildRig(source: THREE.Object3D) {
  const root = clone(source);
  root.updateMatrixWorld(true);

  // Bone names are suffixed with a node index, e.g. "FACIAL_LOD1_C_Jaw_478".
  // The body and face use separate skeletons that both contain neck/head/chest.
  const byName = new Map<string, THREE.Object3D[]>();
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) {
      o.frustumCulled = false;
      o.castShadow = true;
      o.receiveShadow = true;
    }
    const key = o.name.replace(/_\d+$/, "");
    byName.set(key, [...(byName.get(key) ?? []), o]);
  });
  const all = (name: string) => (byName.get(name) ?? []).map(makeJoint);
  const one = (name: string) => all(name)[0];
  const sides = (name: string) => [one(name.replace("*", "L")), one(name.replace("*", "R"))];

  return {
    root,
    heads: all("head"),
    necks: all("neck"),
    chests: all("chest"),
    jaw: one("FACIAL_LOD1_C_Jaw"),
    lipsLower: [
      one("FACIAL_LOD1_C_LipLower"),
      ...sides("FACIAL_LOD1_*_LipLower"),
      ...sides("FACIAL_LOD1_*_LipLowerOuter"),
    ],
    lipsUpper: [
      one("FACIAL_LOD1_C_LipUpper"),
      ...sides("FACIAL_LOD1_*_LipUpper"),
      ...sides("FACIAL_LOD1_*_LipUpperOuter"),
    ],
    lipCorners: sides("FACIAL_LOD1_*_LipCorner"),
    lidsUpper: sides("FACIAL_LOD1_*_EyelidUpper"),
    lidsLower: sides("FACIAL_LOD1_*_EyelidLower"),
    eyeballs: sides("FACIAL_LOD1_*_Eyeball"),
    brows: [...sides("FACIAL_LOD1_*_ForeheadIn"), ...sides("FACIAL_LOD1_*_ForeheadMid")],
    upperArms: [one("LF_uparm"), one("RT_uparm")],
    lowerArms: [one("LF_loarm"), one("RT_loarm")],
  };
}

type Rig = ReturnType<typeof buildRig>;

/** Index 0 = left, 1 = right for the per-side joints; centre joints use both sides averaged. */
const LOWER_LIP_SIDE = [-1, 0, 1, 0, 1];
const UPPER_LIP_SIDE = LOWER_LIP_SIDE;

/**
 * Maps ARKit blendshape weights from Audio2Face-3D onto this model's facial
 * bones. The model has no blendshapes, so each shape is approximated by
 * moving the jaw, lip, corner and brow joints (world-space metres/radians).
 */
function applyBlendshapes(rig: Rig, w: FaceWeights) {
  const W = (k: string) => w[k] ?? 0;
  const side = (base: string, i: number) =>
    i < 0 ? (W(base + "left") + W(base + "right")) / 2 : W(base + (i === 0 ? "left" : "right"));

  const jawOpen = W("jawopen");
  rotate(rig.jaw, jawOpen * 0.3, (W("jawleft") - W("jawright")) * 0.12);
  // Whole-mouth shift toward the character's left (+X) or right.
  const shift = (W("mouthleft") - W("mouthright")) * 0.006;
  const forward = (W("mouthfunnel") + W("mouthpucker")) * 0.006;
  const close = W("mouthclose");

  rig.lipsLower.forEach((j, n) => {
    const i = LOWER_LIP_SIDE[n];
    offset(
      j,
      shift,
      -jawOpen * 0.015 + close * 0.012 - side("mouthlowerdown", i) * 0.005 + W("mouthshruglower") * 0.003,
      forward - W("mouthrolllower") * 0.004 - jawOpen * 0.004
    );
  });
  rig.lipsUpper.forEach((j, n) => {
    const i = UPPER_LIP_SIDE[n];
    offset(
      j,
      shift,
      side("mouthupperup", i) * 0.005 + W("mouthshrugupper") * 0.002,
      forward - W("mouthrollupper") * 0.003
    );
  });
  rig.lipCorners.forEach((j, i) => {
    const out = i === 0 ? 1 : -1; // +X is the character's left
    const smile = side("mouthsmile", i);
    const width = smile * 0.005 + side("mouthstretch", i) * 0.006 - W("mouthpucker") * 0.006 - W("mouthfunnel") * 0.004;
    offset(
      j,
      out * width + shift,
      smile * 0.006 - side("mouthfrown", i) * 0.005 - jawOpen * 0.006,
      -smile * 0.002 - side("mouthdimple", i) * 0.002
    );
  });

  // brows = [L inner, R inner, L mid, R mid]
  const browDown = [W("browdownleft"), W("browdownright")];
  const outerUp = [W("browouterupleft"), W("browouterupright")];
  rig.brows.forEach((j, n) => {
    const i = n % 2;
    const up = n < 2 ? W("browinnerup") * 0.006 : outerUp[i] * 0.005;
    offset(j, 0, up - browDown[i] * 0.004, 0);
  });
}

export default function StanLee({ talking, wordPulse, face }: StanLeeProps) {
  const { scene } = useGLTF(MODEL_URL);
  const rig = useMemo(() => buildRig(scene), [scene]);
  // useFrame mutates the rig every frame, so read it through a ref.
  const rigRef = useRef<ReturnType<typeof buildRig>>(null);
  useEffect(() => {
    rigRef.current = rig;
  }, [rig]);

  const state = useRef({
    open: 0,
    talk: 0,
    nextBlink: 2,
    blinkStart: -1,
    lastPulse: 0,
    pulseTime: -10,
    gaze: new THREE.Vector2(),
    gazeTarget: new THREE.Vector2(),
    nextSaccade: 1,
  });

  useFrame(({ clock }, delta) => {
    const rig = rigRef.current;
    if (!rig) return;
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

    const w = face?.();
    if (w) {
      applyBlendshapes(rig, w);
    } else {
      // Jaw opens, lower lip follows it down, upper lip lifts a touch and the
      // corners pull in slightly.
      const o = s.open;
      rotate(rig.jaw, o * 0.22);
      rig.lipsLower.forEach((j) => offset(j, 0, -o * 0.011, -o * 0.003));
      rig.lipsUpper.forEach((j) => offset(j, 0, o * 0.002, 0));
      rig.lipCorners.forEach((j, i) => offset(j, (i === 0 ? -1 : 1) * o * 0.003, 0, 0));
    }

    // Head motion: gentle idle sway plus nods while talking.
    const turn = Math.sin(t * 0.5) * 0.12 + Math.sin(t * 1.3) * 0.04 * s.talk;
    const nod = Math.sin(t * 1.7) * 0.05 * s.talk + Math.sin(t * 0.4) * 0.015;
    const tilt = Math.sin(t * 0.8) * 0.03;
    rig.necks.forEach((j) => rotate(j, nod * 0.4, turn * 0.4, tilt * 0.4));
    rig.heads.forEach((j) => rotate(j, nod * 0.6, turn * 0.6, tilt * 0.6));

    // Eyebrows lift with emphasis (Audio2Face drives them itself).
    if (!w) {
      const browLift = s.talk * Math.max(0, Math.sin(t * 1.1)) * 0.004 + pulse * 0.002;
      rig.brows.forEach((j) => offset(j, 0, browLift, 0));
    }

    // Blinking.
    if (t > s.nextBlink) {
      s.blinkStart = t;
      s.nextBlink = t + 2 + Math.random() * 3.5;
    }
    const bt = (t - s.blinkStart) / 0.15;
    const blink = bt >= 0 && bt <= 1 ? Math.sin(bt * Math.PI) : 0;
    const lids = [w ? [w.eyeblinkleft, w.eyewideleft, w.eyesquintleft] : [], w ? [w.eyeblinkright, w.eyewideright, w.eyesquintright] : []];
    rig.lidsUpper.forEach((j, i) => {
      const [a2fBlink = 0, wide = 0] = lids[i];
      rotate(j, Math.max(blink, a2fBlink) * 0.55 - wide * 0.15);
    });
    rig.lidsLower.forEach((j, i) => {
      const [a2fBlink = 0, , squint = 0] = lids[i];
      rotate(j, -Math.max(blink, a2fBlink) * 0.15 - squint * 0.1);
    });

    // Small eye saccades.
    if (t > s.nextSaccade) {
      s.gazeTarget.set((Math.random() - 0.5) * 0.12, (Math.random() - 0.5) * 0.06);
      s.nextSaccade = t + 0.6 + Math.random() * 2;
    }
    s.gaze.x = THREE.MathUtils.damp(s.gaze.x, s.gazeTarget.x, 25, delta);
    s.gaze.y = THREE.MathUtils.damp(s.gaze.y, s.gazeTarget.y, 25, delta);
    rig.eyeballs.forEach((j) => rotate(j, -s.gaze.y, s.gaze.x));

    // Breathing.
    rig.chests.forEach((j) => rotate(j, -Math.sin(t * 1.6) * 0.012));

    // Arms: lowered from the A-pose to the sides, with hand gestures while talking.
    const g = s.talk;
    rig.upperArms.forEach((j, i) => {
      const side = i === 0 ? -1 : 1;
      const swing = Math.sin(t * 1.6 + i * 1.3);
      rotate(j, -g * (0.25 + swing * 0.1), 0, side * (0.72 - g * 0.08));
    });
    rig.lowerArms.forEach((j, i) => {
      const swing = Math.sin(t * 2.2 + i * 1.7);
      rotate(j, -0.15 - g * (0.55 + swing * 0.2));
    });
  });

  return <primitive object={rig.root} />;
}

useGLTF.preload(MODEL_URL);
