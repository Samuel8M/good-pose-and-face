import { has, knows, OBJECT_SCHEMA_PREDICATES, sameRole } from './knowledge';
import { LAWS, type Law } from './physics';
import { expr, keyOf, match, similarity, type Description, type Expr, type Term } from './sme';

/*
 * Memory for "what else changes when I swap this?", modelled on Complementary Learning Systems
 * (McClelland, McNaughton & O'Reilly, 1995; Kumaran, Hassabis & McClelland, 2016):
 *
 *  - Episodic memory (fast, hippocampus-like): every choice someone makes is stored at once as a
 *    structured description (entities, attributes, relations; see sme.ts).
 *  - Semantic memory (slow, cortex-like): generalizations built offline by *dreaming*. Replay of
 *    episodes merges analogous ones into probabilistic generalizations, as SAGE does (Kuehne et
 *    al., 2000; McLure et al., 2010), the way non-REM replay abstracts rules; then each
 *    generalization is mapped onto the physical laws (physics.ts) to explain or contrast it and
 *    is extended to analogous objects, the way REM replay forms new associations (Lewis,
 *    Knoblich & Poe, 2018).
 *  - Prior knowledge: the physical laws themselves, retrieved by structure-mapping.
 *
 * A new situation is judged by retrieving the most similar laws, generalizations and episodes
 * (MAC/FAC: a cheap predicate-overlap filter, then SME) and letting them vote.
 */

export type Decision = 'moves' | 'stays';
export type SituationKind =
  | 'held-by-target'
  | 'carried-by-target'
  | 'seat-of-target'
  | 'ridden-by-target'
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
/** What a verdict rests on: a physical law, or the person's own earlier choices. */
export type Basis = { source: 'built-in' | 'you'; note: string; law?: string };
export type Verdict = { decision: Decision; confidence: number; because: Basis };

/** Stored as plain s-expression strings so memory stays readable in storage. */
type StoredDescription = { name: string; exprs: string[] };
export type Episode = { id: string; description: StoredDescription; decision: Decision; created: number; consolidated: boolean; about: string };
export type Generalization = {
  id: string;
  /** Expression → how many member episodes contained it (SAGE's probabilities). */
  counts: Record<string, number>;
  members: number;
  votes: Record<Decision, number>;
  about: string;
};
export type Memory = { episodes: Episode[]; generalizations: Generalization[]; insights: string[]; dreamedAt?: number };

const STORAGE_KEY = 'good-pose-and-face:memory:v2';
const ASSIMILATE = 0.7; // SAGE assimilation threshold on normalised SME similarity
// Remembered choices only vote when structurally close (same kind of relation), so a lesson
// about seats never decides what happens to a held cup.
const RECALL = 0.6;
const GENERIC = new Set(['moves', 'person', 'object']);
const P = 'P', X = 'X';

/** A situation as a structured description: the swapped thing P and the related thing X. */
export function describe(situation: Situation): Description {
  const exprs: Expr[] = [expr('moves', P)];
  const relation: Record<SituationKind, Expr> = {
    'held-by-target': expr('holds', P, X),
    'carried-by-target': expr('carries', P, X),
    'seat-of-target': expr('sits-on', P, X),
    'ridden-by-target': expr('rides', P, X),
    'holder-of-target': expr('holds', X, P),
    'in-front-of-target': expr('in-front-of', X, P),
    'behind-target': expr('behind', X, P),
    'touching-target': expr('touches', X, P),
    'overlapping-target': expr('overlaps-new', P, X),
  };
  const core = relation[situation.relation];
  exprs.push(core);
  if (situation.keptInOther) exprs.push(expr('persists', core));
  exprs.push(expr(situation.kind, X));
  exprs.push(expr(`is-${situation.label.replace(/\s+/g, '-')}`, X));
  if (situation.sizeRatio < 0.15) exprs.push(expr('small', X));
  if (situation.sizeRatio > 0.6) exprs.push(expr('large', X));
  // What the thing affords (object-knowledge.json): lets lessons transfer between analogues.
  if (knows(situation.label)) for (const [schema, pred] of OBJECT_SCHEMA_PREDICATES) if (has(situation.label, schema)) exprs.push(expr(pred, X));
  return { name: 'situation', exprs };
}

const store = (d: Description): StoredDescription => ({ name: d.name, exprs: d.exprs.map(keyOf) });
function parseTerm(text: string): Term {
  const tokens = text.replace(/\(/g, ' ( ').replace(/\)/g, ' ) ').trim().split(/\s+/);
  let i = 0;
  const read = (): Term => {
    const token = tokens[i++];
    if (token !== '(') return token;
    const pred = tokens[i++];
    const args: Term[] = [];
    while (tokens[i] !== ')') args.push(read());
    i += 1;
    return { pred, args };
  };
  return read();
}
const restore = (d: StoredDescription): Description => ({ name: d.name, exprs: d.exprs.map((e) => parseTerm(e) as Expr) });
/** A generalization as a description: the statements most of its members share (p ≥ ½). */
const generalized = (g: Generalization): Description => ({
  name: g.id,
  exprs: Object.entries(g.counts).filter(([, n]) => n / g.members >= 0.5).map(([e]) => parseTerm(e) as Expr),
});

export function loadMemory(): Memory {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? (JSON.parse(raw) as Memory) : null;
    if (parsed && Array.isArray(parsed.episodes) && Array.isArray(parsed.generalizations)) return { ...parsed, insights: parsed.insights ?? [] };
  } catch {
    // Unreadable or blocked storage: start fresh.
  }
  return { episodes: [], generalizations: [], insights: [] };
}

function save(memory: Memory) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(memory));
  } catch {
    // Private windows can refuse storage; memory still works for this visit.
  }
  return memory;
}

/** What a law predicts for X once the situation is mapped onto it (its candidate inferences). */
function lawDecision(lawItem: Law, target: Description): { decision: Decision; sim: number } | null {
  const mapping = match(lawItem.description, target);
  if (mapping.entities.get(X) !== X || mapping.entities.get(P) !== P) return null;
  const inferred = mapping.inferences.find((e) => (e.pred === 'moves-with' || e.pred === 'stays-put') && e.args[0] === X);
  if (!inferred) return null;
  return { decision: inferred.pred === 'moves-with' ? 'moves' : 'stays', sim: similarity(lawItem.description, target, mapping) };
}

const walk = (e: Expr): string[] => [e.pred, ...e.args.flatMap((a) => (typeof a === 'string' ? [] : walk(a)))];
const predicates = (d: Description) => new Set(d.exprs.flatMap(walk));

export function predict(situation: Situation, memory: Memory): Verdict {
  const target = describe(situation);
  const targetPreds = predicates(target);
  type Vote = { decision: Decision; sim: number; weight: number; basis: Basis };
  const votes: Vote[] = [];
  for (const item of LAWS) {
    const result = lawDecision(item, target);
    if (result && result.sim >= 0.3) votes.push({ ...result, weight: 1, basis: { source: 'built-in', note: item.plain, law: item.name } });
  }
  // MAC: only memories sharing some relational vocabulary get the costlier SME comparison.
  const overlaps = (d: Description) => [...predicates(d)].filter((p) => !GENERIC.has(p) && targetPreds.has(p)).length >= 2;
  for (const g of memory.generalizations) {
    const d = generalized(g);
    if (!overlaps(d)) continue;
    const decision: Decision = g.votes.moves >= g.votes.stays ? 'moves' : 'stays';
    const sim = similarity(d, target);
    if (sim < RECALL) continue;
    votes.push({ decision, sim, weight: 3 + Math.min(4, g.members * 0.5), basis: { source: 'you', note: `You chose this in ${g.members} similar situations (${g.about}).` } });
  }
  for (const e of memory.episodes.filter((x) => !x.consolidated)) {
    const d = restore(e.description);
    if (!overlaps(d)) continue;
    const sim = similarity(d, target);
    if (sim < RECALL) continue;
    votes.push({ decision: e.decision, sim, weight: 3, basis: { source: 'you', note: `You chose this before for ${e.about}.` } });
  }
  // FAC: the closest few vote, weighted by structural similarity.
  const top = votes.sort((a, b) => b.sim * b.weight - a.sim * a.weight).slice(0, 5);
  if (!top.length) return { decision: 'stays', confidence: 0.5, because: { source: 'built-in', note: 'Nothing moves it, so it stays (object permanence).' } };
  const tally = { moves: 0, stays: 0 };
  for (const v of top) tally[v.decision] += v.sim * v.sim * v.weight;
  const decision: Decision = tally.moves > tally.stays ? 'moves' : 'stays';
  const winner = top.find((v) => v.decision === decision)!;
  return { decision, confidence: tally[decision] / (tally.moves + tally.stays || 1), because: winner.basis };
}

/** Stores a choice as a new episode (fast, one-shot learning). */
export function learn(memory: Memory, situation: Situation, decision: Decision, about: string): Memory {
  const episode: Episode = {
    id: `ep${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
    description: store(describe(situation)),
    decision,
    created: Date.now(),
    consolidated: false,
    about,
  };
  return save({ ...memory, episodes: [...memory.episodes, episode].slice(-300) });
}

export function forget(): Memory {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing stored to remove.
  }
  return { episodes: [], generalizations: [], insights: [] };
}

export const learnedCount = (memory: Memory) => memory.episodes.length;

const nameOf = (d: Description) => {
  const label = d.exprs.find((e) => e.pred.startsWith('is-'))?.pred.slice(3).replace(/-/g, ' ') ?? 'things';
  const relation = d.exprs.find((e) => ['holds', 'carries', 'sits-on', 'rides', 'in-front-of', 'behind', 'touches', 'overlaps-new'].includes(e.pred));
  const how: Record<string, string> = { holds: 'being held', carries: 'being worn', 'sits-on': 'being sat on', rides: 'being ridden', 'in-front-of': 'in front', behind: 'behind', touches: 'touching', 'overlaps-new': 'in the way' };
  return `the ${label}${relation ? ` ${how[relation.pred]}` : ''}`;
};

/**
 * Offline consolidation ("dreaming"). Phase 1, like non-REM replay: replay new episodes and
 * assimilate each into the most similar generalization (or pair it with a similar episode to
 * start one). Phase 2, like REM replay: map each generalization onto the physical laws and onto
 * analogous objects, and record what was found as insights.
 */
export function dream(memory: Memory): Memory {
  const generalizations = memory.generalizations.map((g) => ({ ...g, counts: { ...g.counts }, votes: { ...g.votes } }));
  const episodes = memory.episodes.map((e) => ({ ...e }));
  const absorb = (g: Generalization, e: Episode) => {
    for (const statement of e.description.exprs) g.counts[statement] = (g.counts[statement] ?? 0) + 1;
    g.members += 1;
    g.votes[e.decision] += 1;
    e.consolidated = true;
  };
  for (const e of episodes.filter((x) => !x.consolidated).sort(() => Math.random() - 0.5)) {
    if (e.consolidated) continue; // absorbed earlier in this dream as another episode's partner
    const d = restore(e.description);
    const best = generalizations
      .map((g) => ({ g, sim: similarity(generalized(g), d) }))
      .sort((a, b) => b.sim - a.sim)[0];
    if (best && best.sim >= ASSIMILATE) {
      absorb(best.g, e);
      continue;
    }
    const partner = episodes.find((o) => o !== e && !o.consolidated && similarity(restore(o.description), d) >= ASSIMILATE);
    if (partner) {
      const g: Generalization = { id: `gen${generalizations.length + 1}`, counts: {}, members: 0, votes: { moves: 0, stays: 0 }, about: nameOf(d) };
      absorb(g, e);
      absorb(g, partner);
      generalizations.push(g);
    }
  }

  // Phase 2: explain each generalization (and each lesson not yet generalized) by the physical
  // law it maps onto best, and extend it by analogy to objects that play the same role.
  const insights: string[] = [];
  const reflections = [
    ...generalizations.map((g) => ({ d: generalized(g), decision: (g.votes.moves >= g.votes.stays ? 'moves' : 'stays') as Decision, about: g.about, times: g.members })),
    ...episodes.filter((e) => !e.consolidated).map((e) => { const d = restore(e.description); return { d, decision: e.decision, about: nameOf(d), times: 1 }; }),
  ];
  for (const g of reflections) {
    const { d, decision } = g;
    const explained = LAWS.map((item) => ({ item, result: lawDecision(item, d) }))
      .filter((x) => x.result)
      .sort((a, b) => b.result!.sim - a.result!.sim)[0];
    const label = d.exprs.find((e) => e.pred.startsWith('is-'))?.pred.slice(3).replace(/-/g, ' ');
    const kin = label ? sameRole(label).slice(0, 3) : [];
    const verb = decision === 'moves' ? 'bring along' : 'leave in place';
    const also = kin.length ? ` By analogy this also applies to ${kin.join(', ')}.` : '';
    const times = g.times === 1 ? 'once' : `${g.times} times`;
    if (!explained) insights.push(`You ${verb} ${g.about} (${times}).${also}`);
    else if (explained.result!.decision === decision) insights.push(`You ${verb} ${g.about}, which matches “${explained.item.name}”: ${explained.item.plain}${also}`);
    else insights.push(`You prefer to ${verb} ${g.about} (${times}), unlike “${explained.item.name}”. Your preference wins in similar photos.${also}`);
  }
  return save({ episodes, generalizations, insights, dreamedAt: Date.now() });
}
