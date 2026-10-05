import { clamp, distance, type Face, type Photo, type Point } from './vision';

/*
 * Matching follows structure-mapping (Gentner) and MAC/FAC retrieval (Forbus, Gentner & Law):
 *
 * 1. Correspondence: "who is who" between two photos is an analogy between two scenes. Surface
 *    attributes (hair, skin and clothing colour, face texture, landmark geometry) propose
 *    pairings, but the winning mapping is the one that preserves the most relational structure:
 *    a single camera-shift hypothesis under which the whole group keeps its arrangement
 *    (systematicity), solved one-to-one.
 * 2. Retrieval: MAC cheaply scores every corresponding face on eye openness, sharpness, pose
 *    and scale; FAC then runs the costlier structural comparison only on the top few.
 */

type Shift = { scale: number; dx: number; dy: number };
export type Match = { face: Face; confidence: number };
export type Candidate = {
  face: Face;
  photo: Photo;
  confidence: number;
  quality: number;
  pose: number;
  structure: number;
  score: number;
  checked: boolean;
};

const FAC_SHORTLIST = 3;

function colourSimilarity(a: number[] | null, b: number[] | null) {
  if (!a || !b) return null;
  return Math.exp(-((Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) / 0.16) ** 2));
}

function geometryError(a: number[], b: number[]) {
  return Math.sqrt(a.reduce((sum, value, index) => sum + (value - b[index]) ** 2, 0) / a.length);
}

/** Surface similarity in [0, 1]: do these two faces look like the same person in the same outfit? */
export function attributeSimilarity(a: Face, b: Face) {
  let weighted = 0;
  let weights = 0;
  const add = (value: number | null, weight: number) => {
    if (value === null) return;
    weighted += value * weight;
    weights += weight;
  };
  let correlation = 0;
  for (let i = 0; i < a.texture.length; i += 1) correlation += a.texture[i] * b.texture[i];
  add(clamp((correlation - 0.2) / 0.7), 0.3);
  add(colourSimilarity(a.context.clothes, b.context.clothes), 0.3);
  add(colourSimilarity(a.context.hair, b.context.hair), 0.2);
  add(colourSimilarity(a.context.skin, b.context.skin), 0.1);
  add(Math.exp(-geometryError(a.signature, b.signature) / 0.12), 0.1);
  return weighted / Math.max(0.0001, weights);
}

function apply(shift: Shift, point: Point) {
  return { x: point.x * shift.scale + shift.dx, y: point.y * shift.scale + shift.dy };
}

function pairScore(shift: Shift, a: Face, b: Face, similarity: number) {
  const expected = a.size * shift.scale;
  const proximity = Math.exp(-((distance(apply(shift, a.center), b.center) / (0.6 * expected)) ** 2));
  const sizeAgreement = Math.exp(-2 * Math.abs(Math.log(b.size / Math.max(1, expected))));
  return proximity * sizeAgreement * (0.35 + 0.65 * similarity);
}

function assign(shift: Shift, from: Face[], to: Face[], similarity: number[][]) {
  const pairs: { i: number; j: number; score: number }[] = [];
  from.forEach((a, i) => to.forEach((b, j) => pairs.push({ i, j, score: pairScore(shift, a, b, similarity[i][j]) })));
  pairs.sort((x, y) => y.score - x.score);
  const usedFrom = new Set<number>();
  const usedTo = new Set<number>();
  const chosen: typeof pairs = [];
  for (const pair of pairs) {
    if (usedFrom.has(pair.i) || usedTo.has(pair.j) || pair.score < 0.05) continue;
    usedFrom.add(pair.i);
    usedTo.add(pair.j);
    chosen.push(pair);
  }
  return { chosen, total: chosen.reduce((sum, pair) => sum + pair.score, 0) };
}

/** For each face in `main`, the same person in `other` (if any). */
export function correspond(main: Photo, other: Photo): Map<string, Match> {
  const result = new Map<string, Match>();
  if (!main.faces.length || !other.faces.length) return result;
  const similarity = main.faces.map((a) => other.faces.map((b) => attributeSimilarity(a, b)));
  // Each candidate pairing proposes a camera shift; keep the one the whole group agrees with.
  const hypotheses: Shift[] = [{ scale: other.width / main.width, dx: 0, dy: 0 }];
  main.faces.forEach((a) => other.faces.forEach((b) => {
    const scale = b.size / Math.max(1, a.size);
    hypotheses.push({ scale, dx: b.center.x - a.center.x * scale, dy: b.center.y - a.center.y * scale });
  }));
  let best = assign(hypotheses[0], main.faces, other.faces, similarity);
  for (const shift of hypotheses.slice(1)) {
    const attempt = assign(shift, main.faces, other.faces, similarity);
    if (attempt.total > best.total) best = attempt;
  }
  for (const pair of best.chosen) {
    if (pair.score < 0.22) continue;
    result.set(main.faces[pair.i].id, { face: other.faces[pair.j], confidence: clamp(pair.score) });
  }
  return result;
}

export function quality(face: Face) {
  return 0.65 * face.eyesOpen + 0.2 * face.sharpness + 0.15 * face.smile;
}

function poseFit(target: Face, donor: Face) {
  return Math.exp(-0.5 * (((target.yaw - donor.yaw) / 0.16) ** 2 + ((target.pitch - donor.pitch) / 0.2) ** 2 + ((target.roll - donor.roll) / 0.24) ** 2));
}

/** Ranked replacement faces for `target`, best first. */
export function rankCandidates(target: Face, photos: Photo[], matches: Map<string, Map<string, Match>>): Candidate[] {
  const pool = photos.flatMap((photo) => {
    const match = matches.get(photo.id)?.get(target.id);
    return match ? [{ photo, ...match }] : [];
  });
  const surfaced = pool
    .map(({ photo, face, confidence }) => {
      const pose = poseFit(target, face);
      // Enlarging a smaller face blurs it; shrinking a larger one is harmless.
      const scale = face.size >= target.size ? 1 : Math.exp(-Math.abs(Math.log(face.size / target.size)) * 0.6);
      const mac = 0.45 * quality(face) + 0.35 * pose + 0.2 * scale;
      return { photo, face, confidence, pose, scale, quality: quality(face), mac };
    })
    .sort((a, b) => b.mac - a.mac);
  return surfaced
    .map((candidate, index) => {
      const checked = index < FAC_SHORTLIST;
      const structure = checked ? Math.exp(-geometryError(target.signature, candidate.face.signature) / 0.12) : 0;
      const score = checked
        ? 0.38 * candidate.quality + 0.2 * candidate.pose + 0.2 * structure + 0.1 * candidate.scale + 0.12 * candidate.confidence
        : candidate.mac * 0.8;
      return { ...candidate, structure, score, checked };
    })
    .sort((a, b) => Number(b.checked) - Number(a.checked) || b.score - a.score);
}

/** Below this width the eyes are only a few pixels, so their state is a guess. */
export const eyesReadable = (face: Face) => face.size >= 60;
// Closed eyes read ~0.0 and a smiling squint ~0.6, so the cut sits well below a squint.
export const eyesLookClosed = (face: Face) => eyesReadable(face) && face.eyesOpen < 0.35;

/** The candidate worth swapping in automatically, if the target's eyes look closed and a clearly better take exists. */
export function suggestedFix(target: Face, candidates: Candidate[]) {
  if (!eyesLookClosed(target)) return undefined;
  return candidates.find((candidate) => eyesReadable(candidate.face) && candidate.face.eyesOpen > 0.5 && candidate.face.eyesOpen > target.eyesOpen + 0.3);
}
