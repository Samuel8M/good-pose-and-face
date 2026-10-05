/*
 * Case-based learning for "what else changes when I swap this?".
 *
 * Every decision is a case: a small relational description of how a related thing stands to the
 * thing being swapped (held by it, sat on by it, standing in front of it…) plus the verdict
 * (moves with it / stays put). New situations are judged by analogy to stored cases, MAC/FAC
 * style: a cheap filter on surface features picks candidates, then a structural comparison
 * ranks them and the closest cases vote. Built-in cases encode common sense; every time someone
 * overrides a verdict, that override is stored as a new, heavier case, so the app learns their
 * preferences and applies them to analogous situations later.
 */

export type Decision = 'moves' | 'stays';
export type SituationKind =
  | 'held-by-target'
  | 'carried-by-target'
  | 'seat-of-target'
  | 'holder-of-target'
  | 'in-front-of-target'
  | 'behind-target'
  | 'touching-target'
  | 'overlapping-target';
export type Situation = {
  relation: SituationKind;
  label: string;
  kind: 'person' | 'object';
  /** Related thing's area ÷ swapped thing's area. */
  sizeRatio: number;
  /** In the other photo, does the same relation still hold between the counterparts? */
  keptInOther: boolean;
};
export type Case = { situation: Situation; decision: Decision; weight: number; source: 'built-in' | 'you'; note: string };
export type Verdict = { decision: Decision; confidence: number; because: Case };

const STORAGE_KEY = 'good-pose-and-face:cases:v1';

const seed = (relation: SituationKind, kind: Situation['kind'], decision: Decision, note: string, extra: Partial<Situation> = {}, weight = 1): Case => ({
  situation: { relation, label: kind === 'person' ? 'person' : '*', kind, sizeRatio: kind === 'person' ? 1 : 0.15, keptInOther: decision === 'moves', ...extra },
  decision,
  weight,
  source: 'built-in',
  note,
});

export const BUILT_IN: Case[] = [
  seed('held-by-target', 'object', 'moves', 'Things a person is holding move with them.'),
  seed('held-by-target', 'object', 'moves', 'Things a person is holding move with them.', { keptInOther: false }, 0.7),
  seed('carried-by-target', 'object', 'moves', 'Bags and ties they wear go with them.'),
  seed('seat-of-target', 'object', 'stays', 'Chairs and benches stay where they are.', { sizeRatio: 0.4, keptInOther: true }),
  seed('holder-of-target', 'person', 'stays', 'The person holding it stays as they are.'),
  seed('in-front-of-target', 'person', 'stays', 'Someone standing in front of them stays in front.'),
  seed('in-front-of-target', 'object', 'stays', 'Things in front of them stay in front.'),
  seed('behind-target', 'person', 'stays', 'People behind them are left alone.'),
  seed('behind-target', 'object', 'stays', 'Things behind them are left alone.'),
  seed('touching-target', 'person', 'stays', "Neighbours they're touching are left alone."),
  seed('touching-target', 'object', 'moves', 'Small things touching them that stay with them in the other shot move too.', { sizeRatio: 0.08, keptInOther: true }, 0.6),
  seed('touching-target', 'object', 'stays', 'Larger things they lean on stay put.', { sizeRatio: 0.5, keptInOther: false }, 0.8),
  seed('overlapping-target', 'person', 'stays', 'Other people stay as they are.'),
  seed('overlapping-target', 'object', 'stays', 'Other things stay as they are.'),
];

function storedCases(): Case[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as Case[]).filter((c) => c && c.situation && (c.decision === 'moves' || c.decision === 'stays')) : [];
  } catch {
    return [];
  }
}

export function loadCases(): Case[] {
  return [...BUILT_IN, ...storedCases()];
}

export function learnedCount() {
  return storedCases().length;
}

/** Records someone's override as a new case. Returns the updated library. */
export function learn(situation: Situation, decision: Decision, describe: string): Case[] {
  const mine = storedCases();
  mine.push({ situation, decision, weight: 3, source: 'you', note: `You chose this before for ${describe}.` });
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(mine.slice(-200)));
  } catch {
    // Private windows can refuse storage; the lesson still applies for this visit.
  }
  return [...BUILT_IN, ...mine];
}

export function forget(): Case[] {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing stored to remove.
  }
  return [...BUILT_IN];
}

/** Structural similarity of two situations (FAC): relation first, then what the thing is. */
function similarity(a: Situation, b: Situation) {
  let score = a.relation === b.relation ? 0.55 : 0;
  if (b.label === '*' || a.label === '*') score += a.kind === b.kind ? 0.15 : 0;
  else score += a.label === b.label ? 0.25 : a.kind === b.kind ? 0.1 : 0;
  score += 0.1 * Math.exp(-Math.abs(Math.log(Math.max(0.01, a.sizeRatio) / Math.max(0.01, b.sizeRatio))));
  score += a.keptInOther === b.keptInOther ? 0.1 : 0;
  return score;
}

export function predict(situation: Situation, cases: Case[]): Verdict {
  // MAC: only cases sharing the relation or the kind of thing are worth comparing closely.
  const pool = cases.filter((c) => c.situation.relation === situation.relation || c.situation.label === situation.label);
  const ranked = (pool.length ? pool : cases)
    .map((c) => ({ c, sim: similarity(situation, c.situation) }))
    .sort((x, y) => y.sim * y.c.weight - x.sim * x.c.weight)
    .slice(0, 5);
  const votes = { moves: 0, stays: 0 };
  for (const { c, sim } of ranked) votes[c.decision] += sim * sim * c.weight;
  const decision: Decision = votes.moves > votes.stays ? 'moves' : 'stays';
  const total = votes.moves + votes.stays || 1;
  const because = ranked.find((r) => r.c.decision === decision)?.c ?? ranked[0].c;
  return { decision, confidence: votes[decision] / total, because };
}
