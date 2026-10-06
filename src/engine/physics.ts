import { parse, type Description } from './sme';

/*
 * The physical world model: laws stored as structured base descriptions (see sme.ts) so that a
 * situation in a photo can be *mapped* onto them, and the law's consequence carried over as a
 * candidate inference. Entities: P = the person/thing being swapped, X = something related to it.
 * Each law also states its physics in formula form; docs/WORLD_MODEL.md explains them.
 */

export type Law = {
  id: string;
  name: string;
  /** The physics, in formula form. */
  formula: string;
  /** One-sentence reason shown to people next to a decision. */
  plain: string;
  description: Description;
};

const law = (id: string, name: string, formula: string, plain: string, ...exprs: string[]): Law => ({
  id, name, formula, plain, description: { name: id, exprs: exprs.map(parse) },
});

export const LAWS: Law[] = [
  law('gravity-support', 'Gravity and support',
    'Weight W = m·g (g ≈ 9.81 m/s²). At rest the forces balance, ΣF = 0, so the hand supplies N = W. Take the hand away and nothing balances W: the thing accelerates down at g.',
    'Gravity: only their hand holds it up, so it has to come along with them.',
    '(moves P)', '(holds P X)', '(graspable X)',
    '(cause (holds P X) (supports P X))',
    '(cause (and (supports P X) (moves P)) (moves-with X P))'),
  law('attachment', 'Rigid attachment',
    'Bodies fastened together share one velocity: v_X = v_P.',
    'It is attached to their body, so it moves with them.',
    '(moves P)', '(carries P X)', '(wearable X)',
    '(cause (carries P X) (attached X P))',
    '(cause (and (attached X P) (moves P)) (moves-with X P))'),
  law('rider-system', 'Rider and ride are one system',
    'A rider supported by a ride moves with it: one system with momentum p = (m_P + m_X)·v.',
    'Rider and ride move as one, so it comes along.',
    '(moves P)', '(rides P X)', '(rideable X)',
    '(cause (rides P X) (supports X P))',
    '(cause (and (supports X P) (moves P)) (moves-with X P))'),
  law('seat-on-ground', 'Seats rest on the ground',
    'The floor supplies the seat\'s normal force N = (m_seat + m_person)·g; static friction (up to μ·N) keeps it in place when the person gets up.',
    'Gravity holds the seat on the floor, so it stays where it is.',
    '(moves P)', '(sits-on P X)', '(seat X)', '(rests-on X ground)',
    '(cause (sits-on P X) (supports X P))',
    '(cause (rests-on X ground) (stays-put X))'),
  law('occlusion', 'Nearer things hide farther things',
    'Light travels in straight lines; along each camera ray only the nearest surface (smallest depth Z) is seen.',
    'Whatever is nearer the camera keeps covering them.',
    '(moves P)', '(in-front-of X P)',
    '(cause (in-front-of X P) (occludes X P))',
    '(cause (occludes X P) (stays-put X))'),
  law('inertia-behind', 'Things behind keep their place',
    'Newton\'s first law: a body at rest stays at rest unless a force acts on it. Something behind them is not pushed by their change of pose.',
    'Nothing pushes what is behind them, so that stays where it is.',
    '(moves P)', '(behind X P)',
    '(cause (behind X P) (independent X P))',
    '(cause (independent X P) (stays-put X))'),
  law('own-feet', 'People carry their own weight',
    'Each person is held up by the ground through their own feet (N = m·g), not by their neighbour.',
    'They stand on their own feet, so they stay as they are.',
    '(moves P)', '(touches X P)', '(person X)', '(stands-on X ground)',
    '(cause (stands-on X ground) (stays-put X))'),
  law('kept-contact', 'Contact that persisted moved together',
    'If two things stayed in contact between shots, they shared the same displacement Δx.',
    'It stayed with them in the other shot, so it comes along.',
    '(moves P)', '(touches X P)', '(small X)', '(persists (touches X P))',
    '(cause (persists (touches X P)) (moves-with X P))'),
  law('leaning-support', 'Big things carry their own weight',
    'A large object resting on the floor is supported by the floor, not by whoever leans on it.',
    'It is big and stands on its own, so it stays put.',
    '(moves P)', '(touches X P)', '(large X)', '(rests-on X ground)',
    '(cause (rests-on X ground) (stays-put X))'),
  law('permanence', 'Object permanence',
    'Things keep existing and keep their place unless something moves them.',
    'People and things keep their place unless something moves them.',
    '(moves P)', '(overlaps-new P X)',
    '(cause (overlaps-new P X) (stays-put X))'),
  law('holder', 'The holder is not the held',
    'Swapping what is held changes the held thing only; the holder\'s body is a separate system.',
    'The person holding it is not part of this swap, so they stay.',
    '(moves P)', '(holds X P)',
    '(cause (holds X P) (separate X P))',
    '(cause (separate X P) (stays-put X))'),
];

/* ------------------------------------------------------------------------------------------ *
 * Light. Faces are close to Lambertian: brightness I = ρ·E·max(0, n·l) (Lambert's cosine law).
 * Ramamoorthi & Hanrahan (2001) and Basri & Jacobs (2003) showed diffuse shading under any
 * distant lighting is almost entirely captured by low-order spherical harmonics; the first-order
 * form I(n) ≈ c0 + c·n is fitted from the face mesh's normals, so a face from one shot can be
 * relit to the light falling on the face it replaces (a quotient/ratio-image relight).
 * ------------------------------------------------------------------------------------------ */

export type Vec3 = [number, number, number];
export type Shading = [number, number, number, number];

/** Per-vertex normals of a triangle mesh (averaged face normals), oriented toward the camera. */
export function vertexNormals(points: { x: number; y: number }[], depth: number[], triangles: ArrayLike<number>): Vec3[] {
  const normals: Vec3[] = points.map(() => [0, 0, 0]);
  for (let i = 0; i < triangles.length; i += 3) {
    const [a, b, c] = [triangles[i], triangles[i + 1], triangles[i + 2]];
    const u: Vec3 = [points[b].x - points[a].x, points[b].y - points[a].y, depth[b] - depth[a]];
    const v: Vec3 = [points[c].x - points[a].x, points[c].y - points[a].y, depth[c] - depth[a]];
    let n: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    // MediaPipe depth: smaller z is nearer the camera, so a camera-facing normal has n.z < 0.
    if (n[2] > 0) n = [-n[0], -n[1], -n[2]];
    for (const idx of [a, b, c]) for (let k = 0; k < 3; k += 1) normals[idx][k] += n[k];
  }
  return normals.map((n) => {
    const length = Math.hypot(n[0], n[1], n[2]) || 1;
    return [n[0] / length, n[1] / length, n[2] / length];
  });
}

/** Least-squares fit of I ≈ c0 + c·n over skin samples (with a little ridge for stability). */
export function fitShading(samples: { n: Vec3; lum: number }[]): Shading | null {
  if (samples.length < 20) return null;
  const m = Array.from({ length: 16 }, () => 0);
  const r = [0, 0, 0, 0];
  for (const { n, lum } of samples) {
    const basis = [1, n[0], n[1], n[2]];
    for (let i = 0; i < 4; i += 1) {
      r[i] += basis[i] * lum;
      for (let j = 0; j < 4; j += 1) m[i * 4 + j] += basis[i] * basis[j];
    }
  }
  for (let i = 1; i < 4; i += 1) m[i * 4 + i] += samples.length * 0.01;
  return solve4(m, r);
}

export const shade = (c: Shading, n: Vec3) => c[0] + c[1] * n[0] + c[2] * n[1] + c[3] * n[2];

function solve4(m: number[], r: number[]): Shading | null {
  const a = [0, 1, 2, 3].map((i) => [...m.slice(i * 4, i * 4 + 4), r[i]]);
  for (let col = 0; col < 4; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 4; row += 1) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (Math.abs(a[pivot][col]) < 1e-9) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let row = 0; row < 4; row += 1) {
      if (row === col) continue;
      const f = a[row][col] / a[col][col];
      for (let k = col; k < 5; k += 1) a[row][k] -= f * a[col][k];
    }
  }
  return [a[0][4] / a[0][0], a[1][4] / a[1][1], a[2][4] / a[2][2], a[3][4] / a[3][3]];
}
