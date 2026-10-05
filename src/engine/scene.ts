import { InteractiveSegmenter, ObjectDetector, PoseLandmarker } from '@mediapipe/tasks-vision';
import wasmLoaderPath from '@mediapipe/tasks-vision/vision_wasm_internal.js?url';
import wasmBinaryPath from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url';
import { clamp, distance, FriendlyError, type Box, type Photo, type Point } from './vision';

/*
 * Scene perception: turns a photo into a relational description, the representation that
 * structure-mapping works over. Entities are people and objects, each with an outline (mask);
 * relations say how they stand to each other (holds, carries, sits on, in front of, touches,
 * left of, above). Mapping two photos is then an analogy between two such descriptions.
 */

export type Mask = {
  /** Foreground confidence 0..1, row-major over `box` at `width`×`height` cells. */
  data: Float32Array;
  width: number;
  height: number;
  /** Photo-pixel region the mask grid covers. */
  box: Box;
};
export type Pose = { points: Point[]; visibility: number[] };
export type Entity = {
  id: string;
  photoId: string;
  kind: 'person' | 'object';
  label: string;
  score: number;
  box: Box;
  mask: Mask;
  /** Outline area in photo pixels. */
  area: number;
  center: Point;
  /** Characteristic size in pixels (square root of area). */
  size: number;
  colour: number[];
  pose?: Pose;
  faceId?: string;
  thumb: string;
};
export type RelationType = 'holds' | 'carries' | 'sits-on' | 'in-front-of' | 'touches' | 'left-of' | 'above';
export type Relation = { type: RelationType; a: string; b: string; weight: number };
export type Scene = { entities: Entity[]; relations: Relation[] };

const MODELS = 'https://storage.googleapis.com/mediapipe-models';
const DETECTOR_URL = `${MODELS}/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite`;
const POSE_URL = `${MODELS}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`;
const SEGMENTER_URL = `${MODELS}/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite`;
const CARRIED = new Set(['handbag', 'backpack', 'tie', 'umbrella', 'suitcase']);
const SEATS = new Set(['chair', 'couch', 'bench', 'bed', 'toilet']);
// Body landmark indices (MediaPipe pose).
const NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24, L_KNEE = 25, R_KNEE = 26;
const HANDS = [15, 16, 17, 18, 19, 20, 21, 22];

type Models = { detector: ObjectDetector; pose: PoseLandmarker; segmenter: InteractiveSegmenter };
let modelsPromise: Promise<Models> | undefined;
let nextId = 0;

export function prepareSceneModels() {
  modelsPromise ??= (async () => {
    const wasm = { wasmLoaderPath, wasmBinaryPath };
    const [detector, pose, segmenter] = await Promise.all([
      ObjectDetector.createFromOptions(wasm, {
        baseOptions: { modelAssetPath: DETECTOR_URL, delegate: 'CPU' },
        runningMode: 'IMAGE',
        scoreThreshold: 0.25,
        maxResults: 40,
      }),
      // Pose outlines are off: in tasks-vision 0.10.35 they abort on the CPU delegate. Outlines
      // come from the interactive segmenter instead, which also handles objects.
      PoseLandmarker.createFromOptions(wasm, {
        baseOptions: { modelAssetPath: POSE_URL, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numPoses: 2,
        outputSegmentationMasks: false,
      }),
      InteractiveSegmenter.createFromOptions(wasm, {
        baseOptions: { modelAssetPath: SEGMENTER_URL, delegate: 'CPU' },
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      }),
    ]);
    return { detector, pose, segmenter };
  })().catch((error: unknown) => {
    modelsPromise = undefined;
    console.error(error);
    throw new FriendlyError('The scene finder could not start. Please check your internet connection and try again.');
  });
  return modelsPromise;
}

const nextFrame = () => new Promise((resolve) => setTimeout(resolve, 0));

function crop(bitmap: ImageBitmap, region: Box, maxSide: number) {
  const scale = Math.min(1, maxSide / Math.max(region.w, region.h));
  const element = document.createElement('canvas');
  element.width = Math.max(1, Math.round(region.w * scale));
  element.height = Math.max(1, Math.round(region.h * scale));
  const context = element.getContext('2d', { willReadFrequently: true })!;
  context.imageSmoothingQuality = 'high';
  context.drawImage(bitmap, region.x, region.y, region.w, region.h, 0, 0, element.width, element.height);
  return { element, context };
}

function grow(box: Box, fx: number, fy: number, width: number, height: number): Box {
  const x = Math.max(0, box.x - box.w * fx);
  const y = Math.max(0, box.y - box.h * fy);
  return { x, y, w: Math.min(width, box.x + box.w * (1 + fx)) - x, h: Math.min(height, box.y + box.h * (1 + fy)) - y };
}

function iou(a: Box, b: Box) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return (ix * iy) / Math.max(1, a.w * a.h + b.w * b.h - ix * iy);
}

function overlapOfSmaller(a: Box, b: Box) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return (ix * iy) / Math.max(1, Math.min(a.w * a.h, b.w * b.h));
}

const inside = (p: Point, box: Box) => p.x >= box.x && p.x <= box.x + box.w && p.y >= box.y && p.y <= box.y + box.h;

type Proposal = { label: string; score: number; box: Box };

/** Many are called: whole-photo and tiled object detection, merged per label. */
function proposeObjects(detector: ObjectDetector, photo: Photo): Proposal[] {
  const { width, height, bitmap } = photo;
  const side = Math.min(width, height) * 0.62;
  const regions: Box[] = [{ x: 0, y: 0, w: width, h: height }];
  for (const fy of [0, 1]) for (const fx of [0, 1]) regions.push({ x: (width - side) * fx, y: (height - side) * fy, w: side, h: side });
  const found: Proposal[] = [];
  for (const region of regions) {
    const { element } = crop(bitmap, region, 1024);
    const scale = region.w / element.width;
    for (const detection of detector.detect(element).detections) {
      const category = detection.categories[0];
      const b = detection.boundingBox;
      if (!category || !b) continue;
      const box = { x: region.x + b.originX * scale, y: region.y + b.originY * scale, w: b.width * scale, h: b.height * scale };
      // Tiles cut big things in half; only trust tile detections that sit clear of inner edges.
      const margin = region.w * 0.01;
      if (region.w < width && ((box.x < region.x + margin && region.x > 0) || (box.x + box.w > region.x + region.w - margin && region.x + region.w < width - 1)
        || (box.y < region.y + margin && region.y > 0) || (box.y + box.h > region.y + region.h - margin && region.y + region.h < height - 1))) continue;
      found.push({ label: category.categoryName, score: category.score, box });
    }
  }
  found.sort((a, b) => b.score - a.score);
  const kept: Proposal[] = [];
  for (const proposal of found) {
    const isPerson = proposal.label === 'person';
    // People in a group overlap heavily, so only merge near-identical boxes; a smaller object
    // inside another object's box under a different label (bench vs chair) is the same thing.
    const duplicate = kept.some((other) => isPerson
      ? other.label === 'person' && iou(other.box, proposal.box) > 0.6
      : other.label !== 'person' && (other.label === proposal.label ? overlapOfSmaller(other.box, proposal.box) > 0.6 : iou(other.box, proposal.box) > 0.5));
    if (!duplicate && (isPerson || proposal.score >= 0.33)) kept.push(proposal);
  }
  return kept;
}

/** People the detector missed but whose face was found: assume a body below the face. */
function proposePeopleFromFaces(photo: Photo, linkedFaces: Set<string>): Proposal[] {
  return photo.faces
    .filter((face) => !linkedFaces.has(face.id))
    .map((face) => ({
      label: 'person',
      score: 0.3,
      box: grow({ x: face.box.x - face.box.w * 1.4, y: face.box.y - face.box.h * 0.35, w: face.box.w * 3.8, h: face.box.h * 7 }, 0, 0, photo.width, photo.height),
    }));
}

function findPose(pose: PoseLandmarker, photo: Photo, box: Box): Pose | undefined {
  const region = grow(box, 0.15, 0.08, photo.width, photo.height);
  const { element } = crop(photo.bitmap, region, 512);
  const result = pose.detect(element);
  let best: Pose | undefined;
  let bestDistance = Infinity;
  for (const landmarks of result.landmarks) {
    const points = landmarks.map((p) => ({ x: region.x + p.x * region.w, y: region.y + p.y * region.h }));
    const torso = { x: (points[L_SHOULDER].x + points[R_SHOULDER].x + points[L_HIP].x + points[R_HIP].x) / 4, y: (points[L_SHOULDER].y + points[R_SHOULDER].y + points[L_HIP].y + points[R_HIP].y) / 4 };
    const d = distance(torso, { x: box.x + box.w / 2, y: box.y + box.h / 2 });
    if (d < bestDistance) {
      bestDistance = d;
      best = { points, visibility: landmarks.map((p) => p.visibility ?? 0) };
    }
  }
  result.close();
  return best && bestDistance < Math.max(box.w, box.h) * 0.45 ? best : undefined;
}

/** Outline of one entity from the interactive segmenter, seeded on the thing itself. */
function outline(segmenter: InteractiveSegmenter, photo: Photo, box: Box, seeds: Point[]): { mask: Mask; colour: number[]; area: number } {
  const region = grow(box, 0.12, 0.08, photo.width, photo.height);
  const { element, context } = crop(photo.bitmap, region, 512);
  const toLocal = (p: Point) => ({ x: clamp((p.x - region.x) / region.w), y: clamp((p.y - region.y) / region.h) });
  const roi = seeds.length > 1 ? { scribble: seeds.map(toLocal) } : { keypoint: toLocal(seeds[0]) };
  const result = segmenter.segment(element, roi);
  const confidence = result.confidenceMasks?.[0]?.getAsFloat32Array() ?? new Float32Array(element.width * element.height);
  result.close();
  const w = element.width, h = element.height;
  // Keep only the part inside the detection box (plus a little), so a neighbour in the same
  // colour clothing can't be swallowed into this outline.
  const keep = { x0: ((box.x - region.x) / region.w) * w - w * 0.04, x1: ((box.x + box.w - region.x) / region.w) * w + w * 0.04, y0: ((box.y - region.y) / region.h) * h - h * 0.03, y1: ((box.y + box.h - region.y) / region.h) * h + h * 0.03 };
  const data = new Float32Array(w * h);
  const pixels = context.getImageData(0, 0, w, h).data;
  const colour = [0, 0, 0];
  let weight = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = y * w + x;
      if (x < keep.x0 || x > keep.x1 || y < keep.y0 || y > keep.y1) continue;
      const value = clamp((confidence[i] - 0.2) / 0.6);
      data[i] = value;
      weight += value;
      for (let c = 0; c < 3; c += 1) colour[c] += pixels[i * 4 + c] * value;
    }
  }
  return {
    mask: { data, width: w, height: h, box: region },
    colour: colour.map((value) => value / Math.max(1, weight) / 255),
    area: weight * (region.w / w) * (region.h / h),
  };
}

/** A cut-out preview: the thing itself at full strength, everything around it faded. */
function thumbnail(bitmap: ImageBitmap, box: Box, mask: Mask) {
  const size = 192;
  const side = Math.max(box.w, box.h) * 1.08;
  const left = box.x + box.w / 2 - side / 2, top = box.y + box.h / 2 - side / 2;
  const element = document.createElement('canvas');
  element.width = size;
  element.height = size;
  const context = element.getContext('2d', { willReadFrequently: true })!;
  context.imageSmoothingQuality = 'high';
  context.fillStyle = '#e4e7e0';
  context.fillRect(0, 0, size, size);
  context.drawImage(bitmap, left, top, side, side, 0, 0, size, size);
  const image = context.getImageData(0, 0, size, size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const px = left + ((x + 0.5) / size) * side, py = top + ((y + 0.5) / size) * side;
      const mx = Math.floor(((px - mask.box.x) / mask.box.w) * mask.width), my = Math.floor(((py - mask.box.y) / mask.box.h) * mask.height);
      const inside = mx >= 0 && my >= 0 && mx < mask.width && my < mask.height ? mask.data[my * mask.width + mx] : 0;
      const keep = 0.22 + 0.78 * inside;
      const i = (y * size + x) * 4;
      for (let c = 0; c < 3; c += 1) image.data[i + c] = image.data[i + c] * keep + 236 * (1 - keep);
    }
  }
  context.putImageData(image, 0, 0);
  return element.toDataURL('image/jpeg', 0.86);
}

/** Seeds for a person's outline: a scribble down the visible body. */
function personSeeds(pose: Pose | undefined, box: Box): Point[] {
  if (!pose) return [{ x: box.x + box.w / 2, y: box.y + box.h * 0.4 }];
  const seen = (i: number) => pose.visibility[i] > 0.5;
  const mid = (a: number, b: number) => ({ x: (pose.points[a].x + pose.points[b].x) / 2, y: (pose.points[a].y + pose.points[b].y) / 2 });
  const seeds: Point[] = [];
  if (seen(NOSE)) seeds.push(pose.points[NOSE]);
  if (seen(L_SHOULDER) && seen(R_SHOULDER)) seeds.push(mid(L_SHOULDER, R_SHOULDER));
  if (seen(L_HIP) && seen(R_HIP)) seeds.push(mid(L_HIP, R_HIP));
  if (seen(L_KNEE) && seen(R_KNEE)) seeds.push(mid(L_KNEE, R_KNEE));
  return seeds.length ? seeds.filter((p) => inside(p, box)) : [{ x: box.x + box.w / 2, y: box.y + box.h * 0.4 }];
}

/** Rasterises every mask onto one coarse grid so relations can compare outlines cheaply. */
export function maskGrid(photo: Photo, cells = 192) {
  const scale = cells / Math.max(photo.width, photo.height);
  const width = Math.max(1, Math.round(photo.width * scale));
  const height = Math.max(1, Math.round(photo.height * scale));
  const paint = (mask: Mask) => {
    const grid = new Float32Array(width * height);
    const x0 = Math.floor(mask.box.x * scale), y0 = Math.floor(mask.box.y * scale);
    const x1 = Math.min(width, Math.ceil((mask.box.x + mask.box.w) * scale)), y1 = Math.min(height, Math.ceil((mask.box.y + mask.box.h) * scale));
    for (let gy = Math.max(0, y0); gy < y1; gy += 1) {
      for (let gx = Math.max(0, x0); gx < x1; gx += 1) {
        const mx = Math.floor(((gx + 0.5) / scale - mask.box.x) / mask.box.w * mask.width);
        const my = Math.floor(((gy + 0.5) / scale - mask.box.y) / mask.box.h * mask.height);
        if (mx >= 0 && my >= 0 && mx < mask.width && my < mask.height) grid[gy * width + gx] = mask.data[my * mask.width + mx];
      }
    }
    return grid;
  };
  return { width, height, paint };
}

function dilate(grid: Float32Array, width: number, height: number, radius: number) {
  const out = new Float32Array(grid.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let best = 0;
      for (let dy = -radius; dy <= radius && best < 1; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const sx = x + dx, sy = y + dy;
          if (sx >= 0 && sy >= 0 && sx < width && sy < height) best = Math.max(best, grid[sy * width + sx]);
        }
      }
      out[y * width + x] = best;
    }
  }
  return out;
}

function relate(photo: Photo, entities: Entity[]): Relation[] {
  const relations: Relation[] = [];
  const grid = maskGrid(photo);
  const painted = entities.map((entity) => grid.paint(entity.mask));
  const grown = painted.map((cells) => dilate(cells, grid.width, grid.height, 2));
  const sums = painted.map((cells) => cells.reduce((total, value) => total + value, 0));
  const contact = (i: number, j: number) => {
    let shared = 0;
    for (let k = 0; k < painted[i].length; k += 1) shared += Math.min(grown[i][k], painted[j][k]);
    return shared / Math.max(1, Math.min(sums[i], sums[j]));
  };

  // Ownership (holds / carries / sits-on) is decided per object: each object has one owner at most.
  type Claim = { person: Entity; type: RelationType; strength: number };
  const claims = new Map<string, Claim>();
  const claim = (thing: Entity, next: Claim) => {
    const current = claims.get(thing.id);
    if (!current || next.strength > current.strength) claims.set(thing.id, next);
  };

  entities.forEach((a, i) => entities.forEach((b, j) => {
    if (i >= j) return;
    const touching = contact(i, j);
    // Spatial frame for the analogy: who is beside / above whom (nearby pairs only).
    if (distance(a.center, b.center) < (a.size + b.size) * 1.6) {
      const [left, right] = a.center.x <= b.center.x ? [a, b] : [b, a];
      relations.push({ type: 'left-of', a: left.id, b: right.id, weight: 0.5 });
      if (Math.abs(a.center.y - b.center.y) > (a.box.h + b.box.h) * 0.2) {
        const [upper, lower] = a.center.y < b.center.y ? [a, b] : [b, a];
        relations.push({ type: 'above', a: upper.id, b: lower.id, weight: 0.5 });
      }
    }
    if (touching < 0.02 && overlapOfSmaller(a.box, b.box) < 0.15) return;
    let owned = false;
    for (const [person, thing] of [[a, b], [b, a]] as const) {
      if (person.kind !== 'person' || thing.kind !== 'object') continue;
      const strength = touching + overlapOfSmaller(person.box, thing.box) * 0.5;
      const pose = person.pose;
      if (CARRIED.has(thing.label) && overlapOfSmaller(person.box, thing.box) > 0.5) {
        claim(thing, { person, type: 'carries', strength });
        owned = true;
      } else if (SEATS.has(thing.label) && pose) {
        const seat = thing.box;
        const sitting = [L_HIP, R_HIP].some((h) => {
          const hip = pose.points[h];
          return hip.x > seat.x - seat.w * 0.3 && hip.x < seat.x + seat.w * 1.3 && hip.y > seat.y - seat.h * 0.6 && hip.y < seat.y + seat.h;
        });
        if (sitting) {
          claim(thing, { person, type: 'sits-on', strength });
          owned = true;
        }
      } else if (pose && thing.area < person.area * 0.6
        && HANDS.some((h) => pose.visibility[h] > 0.3 && inside(pose.points[h], grow(thing.box, 0.25, 0.25, photo.width, photo.height)))) {
        claim(thing, { person, type: 'holds', strength: strength + 0.5 });
        owned = true;
      }
    }
    if (owned) return;
    // Occlusion order: of two overlapping things, the one standing lower in the frame is nearer.
    const aBottom = a.box.y + a.box.h, bBottom = b.box.y + b.box.h;
    if (Math.abs(aBottom - bBottom) > photo.height * 0.02 && (touching > 0.05 || overlapOfSmaller(a.box, b.box) > 0.25)) {
      const [front, back] = aBottom > bBottom ? [a, b] : [b, a];
      relations.push({ type: 'in-front-of', a: front.id, b: back.id, weight: clamp(0.4 + touching) });
    } else if (touching > 0.02) {
      relations.push({ type: 'touches', a: a.id, b: b.id, weight: clamp(0.3 + touching) });
    }
  }));
  for (const [thingId, { person, type, strength }] of claims) {
    relations.push({ type, a: person.id, b: thingId, weight: clamp(0.6 + strength * 0.4) });
  }
  return relations;
}

/** Pairs faces with bodies across the whole photo at once, closest head first. */
function linkFaces(photo: Photo, people: Entity[]) {
  const pairs: { person: Entity; faceId: string; cost: number }[] = [];
  for (const person of people) {
    const pose = person.pose;
    const head = pose && pose.visibility[NOSE] > 0.4 ? pose.points[NOSE] : { x: person.center.x, y: person.box.y + person.box.h * 0.12 };
    for (const face of photo.faces) {
      if (!inside(face.center, grow(person.box, 0.05, 0.05, photo.width, photo.height))) continue;
      const cost = distance(face.center, head) / face.size;
      if (cost < 1.6) pairs.push({ person, faceId: face.id, cost });
    }
  }
  pairs.sort((a, b) => a.cost - b.cost);
  const used = new Set<string>();
  for (const pair of pairs) {
    if (pair.person.faceId || used.has(pair.faceId)) continue;
    pair.person.faceId = pair.faceId;
    used.add(pair.faceId);
  }
  return used;
}

export async function perceiveScene(photo: Photo, onTick: (fraction: number) => void): Promise<Scene> {
  const { detector, pose, segmenter } = await prepareSceneModels();
  const entities: Entity[] = [];
  const build = async (proposals: Proposal[], from: number, to: number) => {
    for (let index = 0; index < proposals.length; index += 1) {
      const proposal = proposals[index];
      const isPerson = proposal.label === 'person';
      const body = isPerson ? findPose(pose, photo, proposal.box) : undefined;
      const seeds = isPerson ? personSeeds(body, proposal.box) : [{ x: proposal.box.x + proposal.box.w / 2, y: proposal.box.y + proposal.box.h / 2 }];
      const { mask, colour, area } = outline(segmenter, photo, proposal.box, seeds);
      if (area >= proposal.box.w * proposal.box.h * 0.08) {
        entities.push({
          id: `e${++nextId}`,
          photoId: photo.id,
          kind: isPerson ? 'person' : 'object',
          label: proposal.label,
          score: proposal.score,
          box: proposal.box,
          mask,
          area,
          center: { x: proposal.box.x + proposal.box.w / 2, y: proposal.box.y + proposal.box.h / 2 },
          size: Math.sqrt(area),
          colour,
          pose: body,
          thumb: thumbnail(photo.bitmap, proposal.box, mask),
        });
      }
      onTick(from + ((index + 1) / proposals.length) * (to - from));
      if (index % 2 === 1) await nextFrame();
    }
  };
  await build(proposeObjects(detector, photo), 0.15, 0.85);
  // Faces that no detected body claims belong to people the detector missed (often someone
  // half-hidden in a group); give them a body too, then link again.
  const linked = linkFaces(photo, entities.filter((e) => e.kind === 'person'));
  await build(proposePeopleFromFaces(photo, linked), 0.85, 0.97);
  linkFaces(photo, entities.filter((e) => e.kind === 'person' && !e.faceId));
  const relations = relate(photo, entities);
  onTick(1);
  return { entities, relations };
}

/** "Person 2", "Teddy bear" … for showing to people. */
export function displayName(entity: Entity, scene: Scene) {
  if (entity.kind === 'person') {
    const people = scene.entities.filter((e) => e.kind === 'person').sort((a, b) => a.center.x - b.center.x);
    return `Person ${people.indexOf(entity) + 1}`;
  }
  const name = entity.label.replace(/_/g, ' ');
  return name.charAt(0).toUpperCase() + name.slice(1);
}
