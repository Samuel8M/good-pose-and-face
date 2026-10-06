// Builds src/engine/object-knowledge.json and docs/OBJECT_ANALOGIES.md from
// research/hico-det-interactions.json plus the curated tables below.
//
//   node scripts/build-object-knowledge.mjs
//
// Every one of the detector's 80 labels gets: its COCO family, what people do with it
// (HICO-DET verbs), the interaction schemas those verbs imply, what happens to it when the
// person it relates to is swapped, a typical real-world size, look-alikes the detector confuses
// it with, and its closest analogues (objects that play the same relational role).

import { readFileSync, writeFileSync } from 'node:fs';
// Node 23.6+ runs TypeScript with type stripping, so the app's own structure-mapping engine is
// used here to compute analogies between objects.
import { keyOf, parse, similarity } from '../src/engine/sme.ts';

const hico = JSON.parse(readFileSync(new URL('../research/hico-det-interactions.json', import.meta.url), 'utf8')).objects;

// COCO super-categories (Lin et al., 2014).
const FAMILIES = {
  person: ['person'],
  vehicle: ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
  outdoor: ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'],
  animal: ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
  accessory: ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'],
  sports: ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'],
  kitchen: ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'],
  food: ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
  furniture: ['chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet'],
  electronic: ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
  appliance: ['microwave', 'oven', 'toaster', 'sink', 'refrigerator'],
  indoor: ['book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
};

// Typical longest visible dimension in metres [low, high]. Common-sense estimates for
// plausibility checks, deliberately wide; not measurements.
const SIZE_M = {
  person: [1.0, 1.95], bicycle: [1.5, 1.9], car: [3.8, 5.0], motorcycle: [1.8, 2.3], airplane: [10, 70], bus: [9, 13], train: [15, 30], truck: [5, 12], boat: [2, 15],
  'traffic light': [0.6, 1.2], 'fire hydrant': [0.6, 0.9], 'stop sign': [0.6, 2.5], 'parking meter': [1.2, 1.6], bench: [1.2, 2.5],
  bird: [0.1, 0.8], cat: [0.3, 0.7], dog: [0.3, 1.1], horse: [1.6, 2.5], sheep: [0.9, 1.4], cow: [1.8, 2.5], elephant: [3, 6.5], bear: [1, 2.5], zebra: [2, 2.6], giraffe: [4, 5.5],
  backpack: [0.35, 0.6], umbrella: [0.6, 1.3], handbag: [0.2, 0.5], tie: [0.3, 0.6], suitcase: [0.5, 0.8],
  frisbee: [0.2, 0.3], skis: [1.5, 2.0], snowboard: [1.3, 1.7], 'sports ball': [0.06, 0.3], kite: [0.5, 2.5], 'baseball bat': [0.75, 0.9], 'baseball glove': [0.25, 0.35], skateboard: [0.7, 0.85], surfboard: [1.7, 3.0], 'tennis racket': [0.6, 0.72],
  bottle: [0.15, 0.4], 'wine glass': [0.13, 0.25], cup: [0.07, 0.16], fork: [0.15, 0.22], knife: [0.15, 0.35], spoon: [0.12, 0.2], bowl: [0.1, 0.35],
  banana: [0.12, 0.25], apple: [0.06, 0.1], sandwich: [0.1, 0.25], orange: [0.06, 0.1], broccoli: [0.1, 0.25], carrot: [0.1, 0.3], 'hot dog': [0.12, 0.25], pizza: [0.2, 0.5], donut: [0.07, 0.13], cake: [0.12, 0.45],
  chair: [0.75, 1.2], couch: [1.4, 2.6], 'potted plant': [0.2, 1.8], bed: [1.9, 2.3], 'dining table': [0.8, 2.8], toilet: [0.55, 0.85],
  tv: [0.5, 1.9], laptop: [0.28, 0.42], mouse: [0.08, 0.13], remote: [0.12, 0.25], keyboard: [0.3, 0.5], 'cell phone': [0.11, 0.18],
  microwave: [0.4, 0.6], oven: [0.55, 0.95], toaster: [0.2, 0.4], sink: [0.4, 1.0], refrigerator: [1.4, 2.0],
  book: [0.12, 0.35], clock: [0.15, 0.6], vase: [0.12, 0.7], scissors: [0.12, 0.25], 'teddy bear': [0.15, 0.9], 'hair drier': [0.18, 0.3], toothbrush: [0.15, 0.2],
};

// Where the thing usually is in a photo.
const PLACEMENT = {
  hand: ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'cell phone', 'remote', 'scissors', 'toothbrush', 'hair drier', 'baseball bat', 'tennis racket', 'frisbee', 'sports ball', 'umbrella', 'book', 'banana', 'apple', 'orange', 'carrot', 'hot dog', 'sandwich', 'donut', 'teddy bear'],
  body: ['tie', 'backpack', 'handbag', 'baseball glove', 'skis', 'snowboard'],
  table: ['bowl', 'pizza', 'cake', 'broccoli', 'laptop', 'keyboard', 'mouse', 'vase', 'clock', 'toaster', 'microwave', 'potted plant'],
  floor: ['chair', 'couch', 'bed', 'dining table', 'toilet', 'bench', 'suitcase', 'refrigerator', 'oven', 'sink', 'tv', 'bicycle', 'motorcycle', 'skateboard', 'dog', 'cat'],
  outdoors: ['car', 'bus', 'train', 'truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
  water: ['boat', 'surfboard'],
  sky: ['airplane', 'kite', 'bird'],
  person: ['person'],
};

// Look-alikes: groups a detector tends to mix up because they share shape and context.
// Used to let the analogy map "chair" in one shot to "bench" in the next, and to re-label an
// implausibly sized detection as a look-alike whose size fits.
const LOOKALIKES = [
  ['chair', 'bench', 'couch', 'toilet', 'bed'],
  ['cup', 'wine glass', 'bottle', 'vase', 'bowl'],
  ['handbag', 'backpack', 'suitcase'],
  ['tv', 'laptop'],
  ['cell phone', 'remote', 'mouse'],
  ['car', 'truck', 'bus'],
  ['bicycle', 'motorcycle'],
  ['skateboard', 'snowboard', 'surfboard', 'skis'],
  ['horse', 'cow', 'zebra'],
  ['sheep', 'dog', 'cow'],
  ['dog', 'cat', 'teddy bear', 'bear'],
  ['apple', 'orange', 'sports ball'],
  ['donut', 'cake', 'pizza'],
  ['sandwich', 'hot dog', 'pizza'],
  ['fork', 'knife', 'spoon', 'toothbrush', 'scissors'],
  ['microwave', 'oven', 'toaster', 'refrigerator'],
  ['stop sign', 'traffic light', 'parking meter'],
  ['baseball bat', 'tennis racket'],
  ['kite', 'bird', 'frisbee'],
  ['dining table', 'bench'],
];

// Schemas: the relational structure between a person and the thing. Each says what happens
// to the thing when the person it is related to is swapped in from another shot.
const SCHEMAS = {
  grasp: { verbs: ['holding', 'carrying', 'eating', 'drinking with', 'sipping', 'swinging', 'throwing', 'catching', 'reading', 'typing on', 'talking on', 'texting on', 'cutting with', 'brushing with', 'hugging', 'wielding', 'licking', 'pointing'], edit: 'moves', why: 'It is in their hands, so it moves with them.' },
  wear: { verbs: ['wearing', 'tying', 'adjusting'], edit: 'moves', why: 'They are wearing it, so it moves with them.' },
  ride: { verbs: ['riding', 'straddling', 'hopping on'], edit: 'moves', why: 'Rider and ride move together between shots.' },
  seat: { verbs: ['sitting on', 'lying on'], edit: 'stays', why: 'Seats and beds stay where they are; the person sits on them.' },
  table: { verbs: ['eating at', 'sitting at'], edit: 'stays', why: 'Tables stay put.' },
  enclose: { verbs: ['boarding', 'exiting', 'driving'], edit: 'stays', why: 'Vehicles they are in or beside stay where they are.' },
  companion: { verbs: ['petting', 'feeding', 'walking', 'grooming', 'hugging', 'kissing'], edit: 'stays', why: 'Animals have their own pose; swap them separately unless they are being carried.' },
  fixture: { verbs: ['operating', 'opening', 'cleaning', 'watching', 'checking', 'repairing', 'controlling', 'flushing', 'paying', 'stopping at', 'standing under'], edit: 'stays', why: 'Fixed things and appliances stay where they are.' },
  play: { verbs: ['kicking', 'dribbling', 'serving', 'hitting', 'blocking', 'flying', 'spinning', 'flipping', 'grinding'], edit: 'stays', why: 'In-play things are mid-air or on the ground; they only move with the person while held.' },
};

const LARGE_VEHICLES = new Set(['car', 'bus', 'train', 'airplane', 'truck', 'boat']);
const ANIMALS = new Set(FAMILIES.animal);
// Animals people actually ride in photos (HICO-DET also lists "straddling a dog").
const MOUNTS = new Set(['horse', 'elephant']);
// Families whose members are fixed in place or plugged in, never picked up in a photo.
const FIXED_FAMILIES = new Set(['outdoor', 'appliance']);
// Curated corrections where verbs alone mislead.
const OVERRIDES = {
  handbag: { add: ['wear'] }, // hangs on the shoulder: worn rather than held in most photos
  suitcase: { add: ['grasp'] },
  umbrella: { add: ['grasp'] },
  kite: { remove: ['grasp'] }, // held by a string far away; the kite itself stays in the sky
  bed: { remove: ['grasp'] },
  couch: { remove: ['grasp'] },
  chair: { remove: ['grasp'] }, // "carrying a chair" is rare in group photos
  refrigerator: { remove: ['grasp'] },
  oven: { remove: ['grasp'] },
  person: { remove: ['grasp'] },
  clock: { remove: ['grasp'], add: ['fixture'] }, // usually on a wall or shelf
  'potted plant': { add: ['fixture'] },
  tv: { add: ['fixture'] },
  dog: { add: ['grasp'] }, // small dogs get carried; the hand relation in the photo decides
};

const familyOf = (label) => Object.entries(FAMILIES).find(([, labels]) => labels.includes(label))[0];
const labels = Object.values(FAMILIES).flat();

const knowledge = {};
for (const label of labels) {
  const verbs = hico[label] ?? [];
  const size = SIZE_M[label];
  const family = familyOf(label);
  const portable = (size[1] <= 1.0 && !FIXED_FAMILIES.has(family)) || ['baseball bat', 'umbrella', 'teddy bear'].includes(label);
  const schemas = new Set();
  for (const [name, schema] of Object.entries(SCHEMAS)) {
    if (!verbs.some((v) => schema.verbs.includes(v))) continue;
    if (name === 'grasp' && !portable) continue;
    if (name === 'ride' && (LARGE_VEHICLES.has(label) || (ANIMALS.has(label) && !MOUNTS.has(label)))) continue;
    if (name === 'wear' && family !== 'accessory' && family !== 'sports') continue;
    if (name === 'play' && family !== 'sports') continue;
    if (name === 'fixture' && (ANIMALS.has(label) || LARGE_VEHICLES.has(label) || family === 'vehicle' || family === 'sports' || label === 'bed' || label === 'dining table')) continue;
    if (name === 'companion' && !ANIMALS.has(label)) continue;
    if (name === 'seat' && (LARGE_VEHICLES.has(label) || ANIMALS.has(label))) continue;
    if (name === 'fixture' && portable) continue;
    schemas.add(name);
  }
  if (LARGE_VEHICLES.has(label)) schemas.add('enclose');
  if (FIXED_FAMILIES.has(family) && !schemas.has('seat')) schemas.add('fixture');
  if (schemas.has('ride')) schemas.delete('seat');
  for (const s of OVERRIDES[label]?.add ?? []) schemas.add(s);
  for (const s of OVERRIDES[label]?.remove ?? []) schemas.delete(s);
  if (label === 'person') schemas.clear();
  const placement = Object.entries(PLACEMENT).find(([, list]) => list.includes(label))?.[0] ?? 'floor';
  knowledge[label] = {
    family,
    animate: label === 'person' || ANIMALS.has(label),
    portable,
    sizeM: size,
    placement,
    schemas: [...schemas],
    verbs,
    lookalikes: [...new Set(LOOKALIKES.filter((g) => g.includes(label)).flat())].filter((l) => l !== label),
  };
}

// Each object as a structured description (structure-mapping representation): attributes of the
// object X, its interactions with a person as relations, and the physics behind its role as
// higher-order causal structure. Analogues are found by SME over these descriptions, so shared
// relational structure (riding, being supported, being held up by a hand) counts far more than
// shared attributes (same family, same size).
const ROLE_STRUCTURE = {
  grasp: ['(holds person X)', '(cause (holds person X) (supports person X))'],
  wear: ['(carries person X)', '(cause (carries person X) (attached X person))'],
  ride: ['(rides person X)', '(cause (rides person X) (supports X person))'],
  seat: ['(sits-on person X)', '(cause (sits-on person X) (supports X person))', '(rests-on X ground)'],
  table: ['(rests-on X ground)', '(cause (rests-on X ground) (stays-put X))'],
  enclose: ['(contains X person)', '(cause (contains X person) (moves-with person X))'],
  companion: ['(agent X)', '(cause (agent X) (moves-by-itself X))'],
  fixture: ['(fixed-to X ground)', '(cause (fixed-to X ground) (stays-put X))'],
  play: ['(launched-by X person)', '(cause (launched-by X person) (airborne X))'],
};
const SCHEMA_ATTRIBUTE = { grasp: 'graspable', wear: 'wearable', ride: 'rideable', seat: 'seat', table: 'table', enclose: 'vehicle', companion: 'animal', fixture: 'fixture', play: 'plaything' };
const sizeClass = (m) => (m < 0.4 ? 'hand-sized' : m < 1.3 ? 'body-sized' : m < 3 ? 'furniture-sized' : 'vehicle-sized');
const slug = (text) => text.replace(/s+/g, '-');
const descriptions = {};
for (const label of labels) {
  const k = knowledge[label];
  const exprs = [
    `(${k.family}-family X)`,
    k.animate ? '(animate X)' : '(inanimate X)',
    k.portable ? '(portable X)' : '(heavy X)',
    `(${sizeClass(k.sizeM[1])} X)`,
    `(found-${k.placement} X)`,
    ...k.schemas.map((schema) => `(${SCHEMA_ATTRIBUTE[schema]} X)`),
    ...k.verbs.map((verb) => `(${slug(verb)} person X)`),
    ...k.schemas.flatMap((schema) => ROLE_STRUCTURE[schema]),
  ];
  descriptions[label] = { name: label, exprs: [...new Set(exprs)].map(parse) };
  knowledge[label].description = descriptions[label].exprs.map(keyOf);
}
for (const label of labels) {
  knowledge[label].analogues = labels
    .filter((other) => other !== label && other !== 'person')
    .map((other) => ({ label: other, score: Math.round(similarity(descriptions[other], descriptions[label]) * 100) / 100 }))
    .filter((a) => a.score >= 0.3)
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
}

const schemaInfo = Object.fromEntries(Object.entries(SCHEMAS).map(([name, s]) => [name, { edit: s.edit, why: s.why, verbs: s.verbs }]));
writeFileSync(new URL('../src/engine/object-knowledge.json', import.meta.url), JSON.stringify({
  about: 'Generated by scripts/build-object-knowledge.mjs. Do not edit by hand; see docs/OBJECT_ANALOGIES.md.',
  sources: [
    'Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014 (labels, super-categories)',
    'Chao et al., "Learning to Detect Human-Object Interactions", WACV 2018 (HICO-DET verbs)',
    'Gentner, "Structure-mapping: A theoretical framework for analogy", Cognitive Science 1983',
    'Gibson, "The Ecological Approach to Visual Perception", 1979 (affordances)',
  ],
  schemas: schemaInfo,
  objects: knowledge,
}, null, 1) + '\n');

// Human-readable report.
const rows = labels.map((l) => {
  const k = knowledge[l];
  return `| ${l} | ${k.family} | ${k.schemas.join(', ') || '—'} | ${k.sizeM[0]}–${k.sizeM[1]} m | ${k.analogues.map((a) => a.label).join(', ') || '—'} | ${k.lookalikes.join(', ') || '—'} |`;
});
const schemaRows = Object.entries(SCHEMAS).map(([name, s]) => `| **${name}** | ${s.edit === 'moves' ? 'Moves with the person' : 'Stays in place'} | ${labels.filter((l) => knowledge[l].schemas.includes(name)).join(', ')} | ${s.why} |`);
const doc = readFileSync(new URL('../docs/OBJECT_ANALOGIES.template.md', import.meta.url), 'utf8')
  .replace('{{SCHEMA_TABLE}}', ['| Schema | When its person is swapped | Members | Why |', '|---|---|---|---|', ...schemaRows].join('\n'))
  .replace('{{OBJECT_TABLE}}', ['| Object | COCO family | Schemas | Typical size | Closest analogues | Look-alikes |', '|---|---|---|---|---|---|', ...rows].join('\n'));
writeFileSync(new URL('../docs/OBJECT_ANALOGIES.md', import.meta.url), doc);
console.log(`Wrote knowledge for ${labels.length} labels.`);
for (const l of ['chair', 'bicycle', 'cup', 'tie', 'dog', 'umbrella', 'tv', 'skis', 'car']) console.log(l.padEnd(10), knowledge[l].schemas.join('+').padEnd(22), '≈', knowledge[l].analogues.map((a) => `${a.label} ${a.score}`).join(', '));
