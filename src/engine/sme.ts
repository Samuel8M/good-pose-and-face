/*
 * A compact Structure-Mapping Engine (after Falkenhainer, Forbus & Gentner, 1989; greedy merge
 * after Forbus & Oblinger, 1990).
 *
 * Knowledge is stored as structured descriptions: entities, attributes (one-place predicates),
 * relations between entities, and higher-order relations between relations, e.g.
 *
 *   (cause (holds P cup) (moves-with cup P))
 *
 * Matching two descriptions finds the largest structurally consistent alignment: identical
 * predicates only, arguments must align too (parallel connectivity), each entity maps to at most
 * one entity (one-to-one), and deeper connected structure scores higher (systematicity).
 * Candidate inferences are base statements carried over to the target: how analogy predicts.
 *
 * This file has no browser dependencies so the knowledge build script can use it under Node.
 */

export type Term = string | Expr;
export type Expr = { pred: string; args: Term[] };
export type Description = { name: string; exprs: Expr[] };
export type Mapping = {
  score: number;
  /** base entity → target entity */
  entities: Map<string, string>;
  /** keys of matched base expressions */
  matched: Set<string>;
  inferences: Expr[];
};

const ATTRIBUTE_WEIGHT = 0.5;
const TRICKLE = 0.8;

export const expr = (pred: string, ...args: Term[]): Expr => ({ pred, args });
export const isExpr = (term: Term): term is Expr => typeof term !== 'string';
export const keyOf = (term: Term): string => (isExpr(term) ? `(${term.pred} ${term.args.map(keyOf).join(' ')})` : term);

/** Every expression in a description, including nested ones, keyed and de-duplicated. */
function allExprs(exprs: Expr[]) {
  const out = new Map<string, Expr>();
  const visit = (e: Expr) => {
    const k = keyOf(e);
    if (out.has(k)) return;
    out.set(k, e);
    for (const a of e.args) if (isExpr(a)) visit(a);
  };
  exprs.forEach(visit);
  return out;
}

type MH = {
  base: Expr;
  target: Expr;
  key: string;
  entities: Map<string, string>;
  children: MH[];
  local: number;
  score: number;
};

/** Structural evaluation of a description against itself (for normalising similarity). */
export function selfScore(description: Description) {
  return match(description, description).score;
}

export function match(base: Description, target: Description): Mapping {
  const baseExprs = allExprs(base.exprs);
  const targetExprs = allExprs(target.exprs);
  const byPred = new Map<string, Expr[]>();
  for (const t of targetExprs.values()) {
    const list = byPred.get(t.pred) ?? [];
    list.push(t);
    byPred.set(t.pred, list);
  }

  // Local match hypotheses, built bottom-up so children exist before parents.
  const hypotheses = new Map<string, MH>();
  const depth = (e: Term): number => (isExpr(e) ? 1 + Math.max(0, ...e.args.map(depth)) : 0);
  const ordered = [...baseExprs.values()].sort((a, b) => depth(a) - depth(b));
  for (const b of ordered) {
    for (const t of byPred.get(b.pred) ?? []) {
      if (t.args.length !== b.args.length) continue;
      const entities = new Map<string, string>();
      const children: MH[] = [];
      let ok = true;
      for (let i = 0; i < b.args.length && ok; i += 1) {
        const ba = b.args[i], ta = t.args[i];
        if (isExpr(ba) !== isExpr(ta)) ok = false;
        else if (!isExpr(ba)) {
          const prior = entities.get(ba as string);
          if (prior !== undefined && prior !== ta) ok = false;
          entities.set(ba as string, ta as string);
        } else {
          const child = hypotheses.get(`${keyOf(ba)}=>${keyOf(ta)}`);
          if (!child) ok = false;
          else {
            children.push(child);
            for (const [be, te] of child.entities) {
              const prior = entities.get(be);
              if (prior !== undefined && prior !== te) ok = false;
              entities.set(be, te);
            }
          }
        }
      }
      // One-to-one inside the hypothesis.
      if (ok && new Set(entities.values()).size !== entities.size) ok = false;
      if (!ok) continue;
      const attribute = b.args.length === 1 && !isExpr(b.args[0]);
      hypotheses.set(`${keyOf(b)}=>${keyOf(t)}`, { base: b, target: t, key: keyOf(b), entities, children, local: attribute ? ATTRIBUTE_WEIGHT : 1, score: 0 });
    }
  }

  // Systematicity: scores trickle down from parents to the structure beneath them.
  const parents = new Map<MH, MH[]>();
  for (const mh of hypotheses.values()) for (const c of mh.children) parents.set(c, [...(parents.get(c) ?? []), mh]);
  const ranked = [...hypotheses.values()].sort((a, b) => depth(b.base) - depth(a.base));
  for (const mh of ranked) {
    const fromParents = (parents.get(mh) ?? []).reduce((best, p) => Math.max(best, p.score), 0);
    mh.score = Math.min(1, mh.local + TRICKLE * fromParents);
  }

  // Kernels: root hypotheses with everything beneath them.
  const roots = [...hypotheses.values()].filter((mh) => !parents.has(mh));
  const kernel = (mh: MH) => {
    const members = new Set<MH>();
    const visit = (m: MH) => { members.add(m); m.children.forEach(visit); };
    visit(mh);
    const total = [...members].reduce((s, m) => s + m.score, 0);
    return { members, entities: mh.entities, total };
  };
  const kernels = roots.map(kernel).sort((a, b) => b.total - a.total);

  // Greedy merge into one consistent global mapping.
  const entities = new Map<string, string>();
  const reverse = new Map<string, string>();
  const usedBase = new Set<string>(), usedTarget = new Set<string>();
  const chosen = new Set<MH>();
  for (const k of kernels) {
    let fits = true;
    for (const [be, te] of k.entities) {
      if ((entities.has(be) && entities.get(be) !== te) || (reverse.has(te) && reverse.get(te) !== be)) fits = false;
    }
    for (const m of k.members) if (!chosen.has(m) && (usedBase.has(m.key) || usedTarget.has(keyOf(m.target)))) fits = false;
    if (!fits) continue;
    for (const [be, te] of k.entities) { entities.set(be, te); reverse.set(te, be); }
    for (const m of k.members) { chosen.add(m); usedBase.add(m.key); usedTarget.add(keyOf(m.target)); }
  }
  const score = [...chosen].reduce((s, m) => s + m.score, 0);

  // Candidate inferences: unmatched base statements that touch the mapping, carried over.
  const inferences: Expr[] = [];
  const project = (term: Term): Term => (isExpr(term) ? { pred: term.pred, args: term.args.map(project) } : entities.get(term) ?? `?${term}`);
  const mentionsMapped = (term: Term): boolean => (isExpr(term) ? term.args.some(mentionsMapped) || usedBase.has(keyOf(term)) : entities.has(term));
  for (const b of base.exprs) {
    if (usedBase.has(keyOf(b)) || !mentionsMapped(b)) continue;
    const visit = (e: Expr) => {
      if (usedBase.has(keyOf(e))) return;
      e.args.forEach((a) => { if (isExpr(a)) visit(a); });
      if (e.args.every((a) => (isExpr(a) ? true : entities.has(a)))) inferences.push(project(e) as Expr);
    };
    visit(b);
  }
  return { score, entities, matched: usedBase, inferences };
}

/** Similarity in [0, 1]: mapping score normalised by both descriptions' self-scores. */
export function similarity(base: Description, target: Description, mapping = match(base, target)) {
  const norm = Math.sqrt(selfScore(base) * selfScore(target)) || 1;
  return Math.min(1, mapping.score / norm);
}

export function parse(text: string): Expr {
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
  return read() as Expr;
}
