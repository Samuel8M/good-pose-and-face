import { predict, type Decision, type Memory, type Situation, type SituationKind, type Verdict } from './memory';
import { isLookalike } from './knowledge';
import type { Match } from './match';
import { maskGrid, type Entity, type RelationType, type Scene } from './scene';
import { clamp, distance, type Face, type Photo, type Point } from './vision';

/*
 * Analogy between two shots of the same moment (structure-mapping, Gentner; SME, Falkenhainer,
 * Forbus & Gentner):
 *
 * 1. Alignment: how the camera moved, estimated from the background so people moving don't
 *    count. This is the frame both descriptions are compared in.
 * 2. Mapping: entities only map to entities of the same kind (identicality: a cup to a cup).
 *    Local "kernel" pairings are scored on position, size and appearance, then re-scored by
 *    systematicity: how many of each pair's relations (holds, sits on, in front of, beside…)
 *    are preserved by the rest of the mapping. A pairing that fits a coherent system of
 *    relations beats an isolated look-alike.
 * 3. Planning an edit: when one entity is swapped for its counterpart, every entity related to
 *    it is judged (moves with it / stays put) by mapping it onto physical laws and remembered
 *    choices (see memory.ts, physics.ts).
 */

/** Camera change: a point p in the main photo is at p·scale + (dx, dy) in the other photo. */
export type Shift = { scale: number; dx: number; dy: number; quality: number };
export type Pair = { entity: Entity; score: number; support: number };
export type Mapping = { shift: Shift; pairs: Map<string, Pair> };
export type Involvement = {
  entity: Entity;
  counterpart?: Entity;
  situation: Situation;
  verdict: Verdict;
  decision: Decision;
  overridden: boolean;
  /** Whether it covers the swapped thing (occluder) rather than sitting behind it. */
  inFront: boolean;
};

export const applyShift = (shift: Shift, p: Point) => ({ x: p.x * shift.scale + shift.dx, y: p.y * shift.scale + shift.dy });

function grayscale(photo: Photo, width: number) {
  const height = Math.max(1, Math.round((photo.height / photo.width) * width));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true })!;
  context.imageSmoothingQuality = 'high';
  context.drawImage(photo.bitmap, 0, 0, width, height);
  const data = context.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < gray.length; i += 1) gray[i] = data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114;
  return { gray, width, height };
}

/** Background weight per cell (people move between shots; walls and floors don't). */
function backgroundWeights(photo: Photo, scene: Scene, width: number, height: number) {
  const grid = maskGrid(photo, Math.max(width, height));
  const weights = new Float32Array(width * height).fill(1);
  for (const entity of scene.entities) {
    if (entity.kind !== 'person') continue;
    const cells = grid.paint(entity.mask);
    for (let i = 0; i < weights.length && i < cells.length; i += 1) weights[i] = Math.min(weights[i], 1 - cells[i]);
  }
  return weights;
}

/** Normalised cross-correlation of main (weighted) against other under a candidate shift. */
function correlation(a: { gray: Float32Array; width: number; height: number }, weights: Float32Array, b: { gray: Float32Array; width: number; height: number }, s: number, dx: number, dy: number, step: number) {
  let sw = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let y = 0; y < a.height; y += step) {
    const by = Math.round(y * s + dy);
    if (by < 0 || by >= b.height) continue;
    for (let x = 0; x < a.width; x += step) {
      const bx = Math.round(x * s + dx);
      if (bx < 0 || bx >= b.width) continue;
      const w = weights[y * a.width + x];
      if (w <= 0.05) continue;
      const va = a.gray[y * a.width + x], vb = b.gray[by * b.width + bx];
      sw += w; sa += w * va; sb += w * vb; saa += w * va * va; sbb += w * vb * vb; sab += w * va * vb;
    }
  }
  if (sw < 50) return -1;
  const cov = sab / sw - (sa / sw) * (sb / sw);
  const varA = saa / sw - (sa / sw) ** 2, varB = sbb / sw - (sb / sw) ** 2;
  return cov / Math.sqrt(Math.max(1e-6, varA * varB));
}

/** Coarse-to-fine search for the camera shift between two shots, using background only. */
export function align(main: Photo, mainScene: Scene, other: Photo): Shift {
  const base = 256;
  const a = grayscale(main, base);
  // Draw the other photo at the same pixels-per-main-pixel so a still camera means scale 1.
  const b = grayscale(other, Math.round(base * (other.width / main.width)));
  const weights = backgroundWeights(main, mainScene, a.width, a.height);
  const s0 = 1;
  let best = { s: s0, dx: 0, dy: 0, score: -2 };
  const range = Math.round(base * 0.18);
  for (let s = s0 * 0.9; s <= s0 * 1.1001; s += 0.025) {
    for (let dy = -range; dy <= range; dy += 4) {
      for (let dx = -range; dx <= range; dx += 4) {
        const score = correlation(a, weights, b, s, dx, dy, 4);
        if (score > best.score) best = { s, dx, dy, score };
      }
    }
  }
  for (const [step, span, sStep] of [[2, 4, 0.01], [1, 2, 0.004], [1, 1, 0.002]] as const) {
    const start = { ...best };
    for (let s = start.s - sStep * 2; s <= start.s + sStep * 2 + 1e-9; s += sStep) {
      for (let dy = start.dy - span; dy <= start.dy + span; dy += 1) {
        for (let dx = start.dx - span; dx <= start.dx + span; dx += 1) {
          const score = correlation(a, weights, b, s, dx, dy, step);
          if (score > best.score) best = { s, dx, dy, score };
        }
      }
    }
  }
  const toPhoto = main.width / base;
  return { scale: best.s, dx: best.dx * toPhoto, dy: best.dy * toPhoto, quality: best.score };
}

function colourSimilarity(a: number[], b: number[]) {
  return Math.exp(-((Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) / 0.18) ** 2));
}

/** Local match quality of two entities before structure is considered. */
function kernel(a: Entity, b: Entity, shift: Shift, faceAgreement: (a: Entity, b: Entity) => number) {
  if (a.kind !== b.kind || (a.kind === 'object' && !isLookalike(a.label, b.label))) return 0;
  // A detector may name the same seat "chair" in one shot and "bench" in the next.
  const naming = a.label === b.label ? 1 : 0.7;
  const expected = applyShift(shift, a.center);
  const reach = 0.5 * Math.max(a.box.w, a.box.h * 0.6) * shift.scale;
  const position = Math.exp(-((distance(expected, b.center) / reach) ** 2));
  const size = Math.exp(-1.5 * Math.abs(Math.log(b.area / Math.max(1, a.area * shift.scale ** 2))));
  return naming * (0.4 * position + 0.15 * size + 0.25 * colourSimilarity(a.colour, b.colour) + 0.2 * faceAgreement(a, b));
}

const key = (type: RelationType, a: string, b: string) => `${type}|${a}|${b}`;

export function mapScenes(main: Photo, mainScene: Scene, other: Photo, otherScene: Scene, faceMatches: Map<string, Match> | undefined): Mapping {
  const shift = align(main, mainScene, other);
  const faceAgreement = (a: Entity, b: Entity) => {
    if (!a.faceId || !b.faceId || !faceMatches) return 0.5;
    const match = faceMatches.get(a.faceId);
    return match ? (match.face.id === b.faceId ? 1 : 0) : 0.5;
  };
  const otherRelations = new Set(otherScene.relations.map((r) => key(r.type, r.a, r.b)));
  const kernels = mainScene.entities.map((a) => otherScene.entities.map((b) => kernel(a, b, shift, faceAgreement)));

  const assign = (scores: number[][]) => {
    const order: { i: number; j: number; score: number }[] = [];
    scores.forEach((row, i) => row.forEach((score, j) => { if (score > 0.25) order.push({ i, j, score }); }));
    order.sort((x, y) => y.score - x.score);
    const usedI = new Set<number>(), usedJ = new Set<number>();
    const chosen = new Map<number, number>();
    for (const { i, j } of order) {
      if (usedI.has(i) || usedJ.has(j)) continue;
      usedI.add(i); usedJ.add(j); chosen.set(i, j);
    }
    return chosen;
  };

  // Systematicity: re-score each pairing by how many of its relations the rest of the mapping preserves.
  let chosen = assign(kernels);
  let support: number[][] = kernels.map((row) => row.map(() => 0));
  for (let round = 0; round < 3; round += 1) {
    const mapped = new Map([...chosen].map(([i, j]) => [mainScene.entities[i].id, otherScene.entities[j].id]));
    support = mainScene.entities.map((a) => otherScene.entities.map((b) => {
      let kept = 0, total = 0;
      for (const r of mainScene.relations) {
        if (r.a !== a.id && r.b !== a.id) continue;
        total += r.weight;
        const partner = r.a === a.id ? mapped.get(r.b) : mapped.get(r.a);
        if (!partner) continue;
        const k = r.a === a.id ? key(r.type, b.id, partner) : key(r.type, partner, b.id);
        if (otherRelations.has(k)) kept += r.weight;
      }
      return total ? kept / total : 0;
    }));
    chosen = assign(kernels.map((row, i) => row.map((k, j) => k * (0.75 + 0.35 * support[i][j]))));
  }
  const pairs = new Map<string, Pair>();
  for (const [i, j] of chosen) {
    pairs.set(mainScene.entities[i].id, { entity: otherScene.entities[j], score: clamp(kernels[i][j] * (0.75 + 0.35 * support[i][j])), support: support[i][j] });
  }
  return { shift, pairs };
}

/** How good a person's take is: open eyes, facing the camera, body visible, sharp. */
export function personQuality(entity: Entity, face: Face | undefined) {
  const eyes = face ? (face.size >= 60 ? face.eyesOpen : 0.7) : 0.5;
  let frontal = 0.5, visible = 0.5;
  if (entity.pose) {
    const { points, visibility } = entity.pose;
    const shoulderWidth = distance(points[11], points[12]);
    const middle = (points[11].x + points[12].x) / 2;
    frontal = shoulderWidth > 1 ? clamp(1 - Math.abs(points[0].x - middle) / (shoulderWidth * 0.6)) : 0.3;
    const body = visibility.slice(11, 29);
    visible = body.reduce((sum, v) => sum + v, 0) / body.length;
  }
  return 0.45 * eyes + 0.25 * frontal + 0.15 * visible + 0.15 * (face?.sharpness ?? 0.5);
}

export type EntityCandidate = { entity: Entity; photo: Photo; score: number; quality: number; mapping: number };

/** Every other shot's version of `target`, best first. */
export function rankEntityCandidates(
  target: Entity,
  others: Photo[],
  mappings: Map<string, Mapping>,
  faceOf: (entity: Entity) => Face | undefined,
): EntityCandidate[] {
  return others
    .flatMap((photo) => {
      const pair = mappings.get(photo.id)?.pairs.get(target.id);
      if (!pair) return [];
      const quality = target.kind === 'person' ? personQuality(pair.entity, faceOf(pair.entity)) : pair.entity.score;
      return [{ entity: pair.entity, photo, quality, mapping: pair.score, score: 0.7 * quality + 0.3 * pair.score }];
    })
    .sort((a, b) => b.score - a.score);
}

function relationBetween(scene: Scene, a: string, b: string) {
  return scene.relations.filter((r) => (r.a === a && r.b === b) || (r.a === b && r.b === a));
}

/** How `other` stands to `target` within one scene, from `other`'s point of view. */
function situationKind(scene: Scene, target: Entity, other: Entity): SituationKind | undefined {
  const relations = relationBetween(scene, target.id, other.id).filter((r) => r.type !== 'left-of' && r.type !== 'above');
  for (const r of relations) {
    if (r.type === 'holds') return r.a === target.id ? 'held-by-target' : 'holder-of-target';
    if (r.type === 'carries') return r.a === target.id ? 'carried-by-target' : 'holder-of-target';
    if (r.type === 'sits-on') return r.a === target.id ? 'seat-of-target' : 'holder-of-target';
    if (r.type === 'rides') return r.a === target.id ? 'ridden-by-target' : 'holder-of-target';
  }
  for (const r of relations) {
    if (r.type === 'in-front-of') return r.a === other.id ? 'in-front-of-target' : 'behind-target';
    if (r.type === 'touches') return 'touching-target';
  }
  return undefined;
}

/**
 * Everything that might be affected by swapping `target` for `donor`, each with a verdict
 * reached by analogy to physical laws and remembered choices (and any override already made).
 */
export function planSwap(
  target: Entity,
  donor: Entity,
  mainScene: Scene,
  otherScene: Scene,
  mapping: Mapping,
  memory: Memory,
  overrides: Record<string, Decision>,
): Involvement[] {
  const reverse = new Map([...mapping.pairs].map(([mainId, pair]) => [pair.entity.id, mainId]));
  // The donor's footprint in main-photo coordinates, to spot things it would now overlap.
  const inverse = (p: Point) => ({ x: (p.x - mapping.shift.dx) / mapping.shift.scale, y: (p.y - mapping.shift.dy) / mapping.shift.scale });
  const corner = inverse({ x: donor.box.x, y: donor.box.y });
  const donorBox = { x: corner.x, y: corner.y, w: donor.box.w / mapping.shift.scale, h: donor.box.h / mapping.shift.scale };
  const overlaps = (e: Entity) => {
    const ix = Math.max(0, Math.min(e.box.x + e.box.w, donorBox.x + donorBox.w) - Math.max(e.box.x, donorBox.x));
    const iy = Math.max(0, Math.min(e.box.y + e.box.h, donorBox.y + donorBox.h) - Math.max(e.box.y, donorBox.y));
    return (ix * iy) / Math.max(1, Math.min(e.box.w * e.box.h, donorBox.w * donorBox.h)) > 0.08;
  };
  const involved: Involvement[] = [];
  for (const entity of mainScene.entities) {
    if (entity.id === target.id) continue;
    const counterpart = mapping.pairs.get(entity.id)?.entity;
    let relation = situationKind(mainScene, target, entity);
    if (!relation && overlaps(entity)) relation = 'overlapping-target';
    if (!relation) continue;
    const otherRelation = counterpart ? situationKind(otherScene, donor, counterpart) : undefined;
    const situation: Situation = {
      relation,
      label: entity.label,
      kind: entity.kind,
      sizeRatio: entity.area / Math.max(1, target.area),
      keptInOther: otherRelation === relation,
    };
    const verdict = predict(situation, memory);
    const override = overrides[entity.id];
    // Depth: in front if it stands in front of the target here, or its counterpart stands in
    // front of the donor in the other shot; whoever holds a swapped object covers it with a hand.
    const inFront = relation === 'in-front-of-target' || relation === 'holder-of-target'
      || (relation === 'overlapping-target' && otherRelation === 'in-front-of-target');
    involved.push({ entity, counterpart, situation, verdict, decision: override ?? verdict.decision, overridden: override !== undefined, inFront });
  }
  // Something that appears only in the donor shot (e.g. a cup now in their hand) comes along.
  for (const thing of otherScene.entities) {
    if (thing.id === donor.id || reverse.has(thing.id) || thing.kind !== 'object') continue;
    const relation = situationKind(otherScene, donor, thing);
    if (relation !== 'held-by-target' && relation !== 'carried-by-target') continue;
    const situation: Situation = { relation, label: thing.label, kind: 'object', sizeRatio: thing.area / Math.max(1, donor.area), keptInOther: true };
    const verdict = predict(situation, memory);
    const override = overrides[thing.id];
    involved.push({ entity: thing, counterpart: thing, situation, verdict, decision: override ?? verdict.decision, overridden: override !== undefined, inFront: false });
  }
  return involved;
}

/** Plain-language reason for a verdict, for showing next to each related thing. */
export function explain(involvement: Involvement, targetName: string) {
  const { situation, decision, overridden, verdict } = involvement;
  const person = situation.kind === 'person';
  const it = person ? 'them' : 'it';
  const its = person ? "they're" : "it's";
  const how: Record<SituationKind, string> = {
    'held-by-target': `${targetName} is holding ${it}`,
    'carried-by-target': `${targetName} is wearing or carrying ${it}`,
    'seat-of-target': `${targetName} is sitting on ${it}`,
    'ridden-by-target': `${targetName} is riding ${it}`,
    'holder-of-target': person ? `they're holding ${targetName}` : `${targetName} rests on it`,
    'in-front-of-target': `${its} in front of ${targetName}`,
    'behind-target': `${its} behind ${targetName}`,
    'touching-target': `${its} touching ${targetName}`,
    'overlapping-target': `the new ${targetName} would overlap ${it}`,
  };
  return {
    verdict: decision === 'moves' ? `Comes along with ${targetName}` : person ? 'Stays exactly as they are' : 'Stays exactly as it is',
    why: `Because ${how[situation.relation]}. ${overridden ? 'You chose this.' : verdict.because.note}`,
  };
}
