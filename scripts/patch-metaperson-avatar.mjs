/**
 * Makes a MetaPerson avatar that went through Sketchfab usable with TalkingHead.
 *
 * Sketchfab's glTF export drops the blend shapes, appends numbers to the bone names
 * (Hips_01, Head_08, ...) and scales the whole model up. MetaPerson avatars share one
 * mesh layout, so the blend shapes (ARKit + Oculus visemes) are copied from another
 * MetaPerson GLB that still has them: vertex by vertex where the meshes match, and by
 * texture coordinate for the head, whose vertex count differs between exporters.
 *
 *   node scripts/patch-metaperson-avatar.mjs <sketchfab.glb> <donor.glb> <out.glb>
 */

import fs from "node:fs";

const [srcPath, donorPath, outPath] = process.argv.slice(2);
if (!outPath) {
  console.error("usage: node scripts/patch-metaperson-avatar.mjs <sketchfab.glb> <donor.glb> <out.glb>");
  process.exit(1);
}

// Donor meshes whose blend shapes are copied. The eyeballs are left out: Sketchfab's
// copies have a separate cornea the donor lacks, and TalkingHead turns the eyes with bones.
const MORPH_MESHES = ["AvatarHead", "AvatarEyelashes", "AvatarTeethLower"];

// ---------------------------------------------------------------- GLB I/O

function readGlb(path) {
  const b = fs.readFileSync(path);
  if (b.readUInt32LE(0) !== 0x46546c67) throw new Error(`${path} is not a GLB file`);
  const jsonLen = b.readUInt32LE(12);
  const json = JSON.parse(b.subarray(20, 20 + jsonLen).toString("utf8"));
  const binStart = 20 + jsonLen;
  const bin = binStart < b.length ? b.subarray(binStart + 8, binStart + 8 + b.readUInt32LE(binStart)) : Buffer.alloc(0);
  return { json, bin };
}

function writeGlb(path, json, bin) {
  const pad = (buf, byte) => Buffer.concat([buf, Buffer.alloc((4 - (buf.length % 4)) % 4, byte)]);
  const jsonBuf = pad(Buffer.from(JSON.stringify(json), "utf8"), 0x20);
  const binBuf = pad(bin, 0);
  const header = Buffer.alloc(12);
  header.writeUInt32LE(0x46546c67, 0);
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(12 + 8 + jsonBuf.length + 8 + binBuf.length, 8);
  const chunk = (buf, type) => {
    const h = Buffer.alloc(8);
    h.writeUInt32LE(buf.length, 0);
    h.writeUInt32LE(type, 4);
    return Buffer.concat([h, buf]);
  };
  fs.writeFileSync(path, Buffer.concat([header, chunk(jsonBuf, 0x4e4f534a), chunk(binBuf, 0x004e4942)]));
}

const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const TYPED = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };

function readView(glb, viewIndex, offset, type, count, componentType) {
  const view = glb.json.bufferViews[viewIndex];
  const Typed = TYPED[componentType];
  const n = COMPONENTS[type];
  const stride = view.byteStride || n * Typed.BYTES_PER_ELEMENT;
  const out = new Float64Array(count * n);
  const base = (view.byteOffset || 0) + (offset || 0);
  const dv = new DataView(glb.bin.buffer, glb.bin.byteOffset, glb.bin.byteLength);
  const get = {
    5121: (o) => dv.getUint8(o),
    5123: (o) => dv.getUint16(o, true),
    5125: (o) => dv.getUint32(o, true),
    5126: (o) => dv.getFloat32(o, true),
  }[componentType];
  for (let i = 0; i < count; i++)
    for (let c = 0; c < n; c++) out[i * n + c] = get(base + i * stride + c * Typed.BYTES_PER_ELEMENT);
  return out;
}

/** Reads an accessor (dense or sparse) into a flat Float64Array. */
function readAccessor(glb, index) {
  const a = glb.json.accessors[index];
  const n = COMPONENTS[a.type];
  const out =
    a.bufferView === undefined
      ? new Float64Array(a.count * n)
      : readView(glb, a.bufferView, a.byteOffset, a.type, a.count, a.componentType);
  if (a.sparse) {
    const s = a.sparse;
    const idx = readView(glb, s.indices.bufferView, s.indices.byteOffset, "SCALAR", s.count, s.indices.componentType);
    const val = readView(glb, s.values.bufferView, s.values.byteOffset, a.type, s.count, a.componentType);
    for (let i = 0; i < s.count; i++) for (let c = 0; c < n; c++) out[idx[i] * n + c] = val[i * n + c];
  }
  return out;
}

/** Appends a sparse VEC3 float accessor holding only the non-zero entries of `deltas`. */
function addSparseVec3(glb, chunks, deltas, count) {
  const idx = [];
  for (let i = 0; i < count; i++) {
    if (Math.abs(deltas[i * 3]) + Math.abs(deltas[i * 3 + 1]) + Math.abs(deltas[i * 3 + 2]) > 1e-7) idx.push(i);
  }
  const min = [0, 0, 0];
  const max = [0, 0, 0];
  const values = new Float32Array(idx.length * 3);
  idx.forEach((v, k) => {
    for (let c = 0; c < 3; c++) {
      const d = deltas[v * 3 + c];
      values[k * 3 + c] = d;
      min[c] = Math.min(min[c], d);
      max[c] = Math.max(max[c], d);
    }
  });
  const accessor = { componentType: 5126, count, type: "VEC3", min, max };
  if (idx.length) {
    const addView = (typed) => {
      let offset = chunks.length ? chunks.reduce((s, c) => s + c.length, 0) : 0;
      offset += glb.bin.length;
      const buf = Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);
      const padded = Buffer.concat([buf, Buffer.alloc((4 - (buf.length % 4)) % 4)]);
      chunks.push(padded);
      glb.json.bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: buf.length });
      return glb.json.bufferViews.length - 1;
    };
    accessor.sparse = {
      count: idx.length,
      indices: { bufferView: addView(Uint32Array.from(idx)), componentType: 5125 },
      values: { bufferView: addView(values) },
    };
  }
  glb.json.accessors.push(accessor);
  return glb.json.accessors.length - 1;
}

// ---------------------------------------------------------------- matrices (column-major 4x4)

function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) out[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return out;
}

function localMatrix(node) {
  if (node.matrix) return node.matrix;
  const [x, y, z, w] = node.rotation || [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale || [1, 1, 1];
  const [tx, ty, tz] = node.translation || [0, 0, 0];
  return [
    (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
    2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
    2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

function worldMatrices(json) {
  const parent = {};
  json.nodes.forEach((n, i) => (n.children || []).forEach((c) => (parent[c] = i)));
  const cache = {};
  const world = (i) =>
    (cache[i] ??= parent[i] === undefined ? localMatrix(json.nodes[i]) : multiply(world(parent[i]), localMatrix(json.nodes[i])));
  return json.nodes.map((_, i) => world(i));
}

// ---------------------------------------------------------------- patch

const src = readGlb(srcPath);
const donor = readGlb(donorPath);
const { json } = src;

// 1. Drop Sketchfab's display transform above the model (a Z-up flip and its inverse,
//    plus a uniform scale), so the avatar is in metres with Y up like other MetaPersons.
const sceneRoots = json.scenes[json.scene ?? 0].nodes;
const before = worldMatrices(json);
for (let i = sceneRoots[0]; i !== undefined; i = json.nodes[i].children?.length === 1 ? json.nodes[i].children[0] : undefined) {
  const n = json.nodes[i];
  if (json.skins.some((s) => s.joints.includes(i)) || n.mesh !== undefined) break;
  delete n.matrix;
  delete n.translation;
  delete n.rotation;
  delete n.scale;
}
const after = worldMatrices(json);

// The inverse bind matrices were computed with that transform in place; take it out of
// them too, so the vertices (already in metres) render where they are.
for (const skin of json.skins) {
  if (skin.inverseBindMatrices === undefined) continue;
  const ibm = readAccessor(src, skin.inverseBindMatrices);
  const meshNode = json.nodes.findIndex((n) => n.skin === json.skins.indexOf(skin));
  const removed = multiply(before[meshNode], invert(after[meshNode]));
  const fixed = new Float32Array(ibm.length);
  skin.joints.forEach((joint, k) => {
    const old = Array.from(ibm.subarray(k * 16, k * 16 + 16));
    const bind = multiply(before[joint], old);
    fixed.set(multiply(multiply(invert(after[joint]), bind), invert(removed)), k * 16);
  });
  const view = json.bufferViews[json.accessors[skin.inverseBindMatrices].bufferView];
  const offset = (view.byteOffset || 0) + (json.accessors[skin.inverseBindMatrices].byteOffset || 0);
  Buffer.from(fixed.buffer).copy(src.bin, offset);
}

function invert(m) {
  // General 4x4 inverse (cofactor expansion).
  const inv = new Array(16);
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  return inv.map((v) => v / det);
}

// 2. Bone names as TalkingHead expects them: Hips_01 -> Hips.
for (const skin of json.skins) {
  for (const joint of skin.joints) {
    const n = json.nodes[joint];
    if (n.name !== "_rootJoint") n.name = n.name.replace(/_\d+$/, "");
  }
}
// TalkingHead's model root: the node holding the meshes and the skeleton.
const skeletonRoot = json.nodes.findIndex((n) => n.name === "_rootJoint");
json.nodes.find((n) => n.children?.includes(skeletonRoot)).name = "Armature";

// 3. Blend shapes from the donor.
const meshInfo = (glb, m) => {
  const prim = glb.json.meshes[m].primitives[0];
  return {
    count: glb.json.accessors[prim.attributes.POSITION].count,
    pos: readAccessor(glb, prim.attributes.POSITION),
    uv: readAccessor(glb, prim.attributes.TEXCOORD_0),
  };
};

/** Sketchfab names meshes "<Name>_<Name>_0", except some it just numbers. */
function findTarget(name, donorCount) {
  const named = json.meshes.findIndex((m) => m.name.startsWith(name + "_") || m.name === name);
  if (named >= 0) return named;
  let best = -1;
  let bestDiff = Infinity;
  json.meshes.forEach((m, i) => {
    if (!/^\d+$/.test(m.name)) return;
    const diff = Math.abs(json.accessors[m.primitives[0].attributes.POSITION].count - donorCount) / donorCount;
    if (diff < bestDiff) [best, bestDiff] = [i, diff];
  });
  return bestDiff < 0.1 ? best : -1;
}

const center = (pos) => {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i++) {
    min[i % 3] = Math.min(min[i % 3], pos[i]);
    max[i % 3] = Math.max(max[i % 3], pos[i]);
  }
  return min.map((v, c) => (v + max[c]) / 2);
};

/**
 * For each target vertex, the donor vertex with the same texture coordinate. Ties (seams,
 * eyelash cards that share UVs) go to the nearest vertex once both meshes are centred.
 * Sketchfab flips V on export, which is detected and undone first.
 */
function correspond(t, d) {
  if (t.count === d.count) {
    let maxDiff = 0;
    for (let i = 0; i < t.uv.length; i++) maxDiff = Math.max(maxDiff, Math.abs(t.uv[i] - d.uv[i]));
    if (maxDiff < 1e-4) return { map: Int32Array.from({ length: t.count }, (_, i) => i), note: "same vertex order" };
  }
  t = { ...t, uv: Float64Array.from(t.uv), pos: Float64Array.from(t.pos) };
  const [tc, dc] = [center(t.pos), center(d.pos)];
  for (let i = 0; i < t.pos.length; i++) t.pos[i] += dc[i % 3] - tc[i % 3];
  const nearestUv = (u, v) => {
    let best = Infinity;
    for (let k = 0; k < d.count; k++) best = Math.min(best, Math.hypot(d.uv[k * 2] - u, d.uv[k * 2 + 1] - v));
    return best;
  };
  let straight = 0;
  let flipped = 0;
  for (let i = 0; i < t.count; i += Math.ceil(t.count / 200)) {
    straight += nearestUv(t.uv[i * 2], t.uv[i * 2 + 1]);
    flipped += nearestUv(t.uv[i * 2], 1 - t.uv[i * 2 + 1]);
  }
  if (flipped < straight) for (let i = 1; i < t.uv.length; i += 2) t.uv[i] = 1 - t.uv[i];

  const CELL = 0.01;
  const grid = new Map();
  for (let i = 0; i < d.count; i++) {
    const key = `${Math.floor(d.uv[i * 2] / CELL)},${Math.floor(d.uv[i * 2 + 1] / CELL)}`;
    (grid.get(key) ?? grid.set(key, []).get(key)).push(i);
  }
  const map = new Int32Array(t.count);
  let sum = 0;
  let worst = 0;
  for (let i = 0; i < t.count; i++) {
    const [u, v] = [t.uv[i * 2], t.uv[i * 2 + 1]];
    const [cu, cv] = [Math.floor(u / CELL), Math.floor(v / CELL)];
    let best = -1;
    let bestScore = Infinity;
    for (let r = 1; best < 0 || r <= 1; r++) {
      for (let a = cu - r; a <= cu + r; a++)
        for (let b = cv - r; b <= cv + r; b++)
          for (const k of grid.get(`${a},${b}`) ?? []) {
            const duv = Math.hypot(d.uv[k * 2] - u, d.uv[k * 2 + 1] - v);
            const dpos = Math.hypot(d.pos[k * 3] - t.pos[i * 3], d.pos[k * 3 + 1] - t.pos[i * 3 + 1], d.pos[k * 3 + 2] - t.pos[i * 3 + 2]);
            const score = duv + dpos * 0.05; // UV decides; position only breaks ties (seams, mirrored islands)
            if (score < bestScore) [best, bestScore] = [k, score];
          }
      if (r > 50) break;
    }
    map[i] = best;
    const duv = Math.hypot(d.uv[best * 2] - u, d.uv[best * 2 + 1] - v);
    sum += duv;
    worst = Math.max(worst, duv);
  }
  return { map, note: `matched by UV, mean ${(sum / t.count).toExponential(1)}, worst ${worst.toExponential(1)}` };
}

src.bin = Buffer.concat([src.bin, Buffer.alloc((4 - (src.bin.length % 4)) % 4)]);
const extra = [];
for (const name of MORPH_MESHES) {
  const dm = donor.json.meshes.findIndex((m) => m.name === name);
  if (dm < 0) throw new Error(`Donor has no mesh ${name}`);
  const dPrim = donor.json.meshes[dm].primitives[0];
  const names = donor.json.meshes[dm].extras?.targetNames ?? [];
  const d = meshInfo(donor, dm);
  const tm = findTarget(name, d.count);
  if (tm < 0) throw new Error(`No mesh in ${srcPath} matches ${name}`);
  const t = meshInfo(src, tm);
  const { map, note } = correspond(t, d);

  const targets = names.map((_, k) => {
    const dd = readAccessor(donor, dPrim.targets[k].POSITION);
    const deltas = new Float64Array(t.count * 3);
    for (let i = 0; i < t.count; i++) for (let c = 0; c < 3; c++) deltas[i * 3 + c] = dd[map[i] * 3 + c];
    return { POSITION: addSparseVec3(src, extra, deltas, t.count) };
  });
  const mesh = json.meshes[tm];
  mesh.name = name;
  for (const prim of mesh.primitives) prim.targets = targets;
  mesh.weights = names.map(() => 0);
  mesh.extras = { ...mesh.extras, targetNames: names };
  console.log(`${name}: ${names.length} blend shapes onto ${t.count} vertices (${note})`);
}

src.bin = Buffer.concat([src.bin, ...extra]);
json.buffers[0].byteLength = src.bin.length;
writeGlb(outPath, json, src.bin);
console.log(`wrote ${outPath} (${(fs.statSync(outPath).size / 1e6).toFixed(1)} MB)`);
