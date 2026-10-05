import { FaceDetector, FaceLandmarker, type Category } from '@mediapipe/tasks-vision';
// Bundle the WASM runtime that ships with the installed package so the loader and
// binary always match the JS API version (a CDN URL pinned to another version 404s).
import wasmLoaderPath from '@mediapipe/tasks-vision/vision_wasm_internal.js?url';
import wasmBinaryPath from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url';

export type Point = { x: number; y: number };
export type Box = { x: number; y: number; w: number; h: number };
export type Face = {
  id: string;
  photoId: string;
  /** 468 mesh landmarks in photo pixels. */
  points: Point[];
  box: Box;
  center: Point;
  /** Face width in pixels; the unit for every distance comparison. */
  size: number;
  eyesOpen: number;
  smile: number;
  sharpness: number;
  yaw: number;
  pitch: number;
  roll: number;
  /** Pose-normalized arrangement of eyes, nose, mouth and chin (structural relations). */
  signature: number[];
  /** Aligned, contrast-normalized grayscale patch of the lower face. */
  texture: Float32Array;
  /** Mean colours of hair, cheeks and clothing: same-session identity context. */
  context: { hair: number[] | null; skin: number[]; clothes: number[] | null };
  thumb: string;
};
export type Photo = {
  id: string;
  file: File;
  name: string;
  bitmap: ImageBitmap;
  width: number;
  height: number;
  thumb: string;
  faces: Face[];
};

export class FriendlyError extends Error {}

const MODELS = 'https://storage.googleapis.com/mediapipe-models';
const DETECTOR_URL = `${MODELS}/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite`;
const LANDMARKER_URL = `${MODELS}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`;
// Faces are found and matched on a copy at most this big; saving re-reads the originals at full size.
const WORKING_SIDE = 4096;
const MESH_SIZE = 468;
export const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

type Models = { detector: FaceDetector; landmarker: FaceLandmarker };
let modelsPromise: Promise<Models> | undefined;
let nextId = 0;
const uid = (prefix: string) => `${prefix}${++nextId}`;

/**
 * Two models: a fast full-range detector that proposes faces of any size, and the detailed
 * landmarker that confirms each proposal from a close crop and maps its 468 points.
 */
export function prepareModels() {
  modelsPromise ??= (async () => {
    const wasm = { wasmLoaderPath, wasmBinaryPath };
    const [detector, landmarker] = await Promise.all([
      FaceDetector.createFromOptions(wasm, {
        baseOptions: { modelAssetPath: DETECTOR_URL, delegate: 'CPU' },
        runningMode: 'IMAGE',
        minDetectionConfidence: 0.35,
      }),
      FaceLandmarker.createFromOptions(wasm, {
        baseOptions: { modelAssetPath: LANDMARKER_URL, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numFaces: 3,
        outputFaceBlendshapes: true,
        minFaceDetectionConfidence: 0.4,
        minFacePresenceConfidence: 0.4,
      }),
    ]);
    return { detector, landmarker };
  })().catch((error: unknown) => {
    // Forget the failure so the next attempt retries instead of failing forever.
    modelsPromise = undefined;
    console.error(error);
    throw new FriendlyError('The face finder could not start. Please check your internet connection and try again.');
  });
  return modelsPromise;
}

export const clamp = (value: number, min = 0, max = 1) => Math.max(min, Math.min(max, value));
export const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a: Point, b: Point) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const nextFrame = () => new Promise((resolve) => setTimeout(resolve, 0));

function canvas(width: number, height: number) {
  const element = document.createElement('canvas');
  element.width = Math.max(1, Math.round(width));
  element.height = Math.max(1, Math.round(height));
  const context = element.getContext('2d', { willReadFrequently: true });
  if (!context) throw new FriendlyError('This browser could not open a drawing surface. Try Chrome, Edge, Safari or Firefox.');
  context.imageSmoothingQuality = 'high';
  return { element, context };
}

async function decode(file: File, maxSide: number): Promise<ImageBitmap> {
  const isHeic = /\.(heic|heif)$/i.test(file.name) || /hei[cf]/i.test(file.type);
  let bitmap: ImageBitmap | undefined;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    // Older engines reject the option; an <img> still honours the camera's rotation tag.
    const url = URL.createObjectURL(file);
    try {
      const image = await new Promise<HTMLImageElement>((resolve, reject) => {
        const element = new Image();
        element.onload = () => resolve(element);
        element.onerror = reject;
        element.src = url;
      });
      const { element, context } = canvas(image.naturalWidth, image.naturalHeight);
      context.drawImage(image, 0, 0);
      bitmap = await createImageBitmap(element);
    } catch {
      bitmap = undefined;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  if (!bitmap) {
    throw new FriendlyError(isHeic
      ? `"${file.name}" is an iPhone HEIC photo, which this browser can't open. Email or AirDrop it to yourself as a JPG, or set the iPhone camera to "Most Compatible".`
      : `"${file.name}" couldn't be opened. Please choose a JPG or PNG photo.`);
  }
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  if (scale < 1) {
    const resized = await createImageBitmap(bitmap, {
      resizeWidth: Math.round(bitmap.width * scale),
      resizeHeight: Math.round(bitmap.height * scale),
      resizeQuality: 'high',
    });
    bitmap.close();
    bitmap = resized;
  }
  return bitmap;
}

/** The photo at its original resolution, for saving. */
export const decodeFull = (file: File) => decode(file, Infinity);

/** Whole image plus overlapping square tiles, so small faces in group shots fill enough of a crop to be found. */
function detectionCrops(width: number, height: number): Box[] {
  const long = Math.max(width, height);
  const crops: Box[] = [{ x: 0, y: 0, w: width, h: height }];
  const grids = long >= 2400 ? [2, 3, 4, 6] : long >= 500 ? [2, 3, 4] : [];
  for (const grid of grids) {
    const side = Math.min(width, height, (2 * long) / (grid + 1));
    const stride = side / 2;
    const columns = Math.max(1, Math.ceil((width - side) / stride - 0.01) + 1);
    const rows = Math.max(1, Math.ceil((height - side) / stride - 0.01) + 1);
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        crops.push({
          x: columns === 1 ? 0 : ((width - side) * column) / (columns - 1),
          y: rows === 1 ? 0 : ((height - side) * row) / (rows - 1),
          w: side,
          h: side,
        });
      }
    }
  }
  return crops;
}

type RawFace = { points: Point[]; blend: Category[]; box: Box };
type Proposal = { box: Box; score: number };

function boundsOf(points: Point[]): Box {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x); maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y); maxY = Math.max(maxY, point.y);
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function overlaps(a: Box, b: Box) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const intersection = ix * iy;
  const smaller = Math.min(a.w * a.h, b.w * b.h);
  return intersection / Math.max(1, smaller) > 0.35;
}

function drawCrop(bitmap: ImageBitmap, crop: Box, side: number) {
  const scale = side / Math.max(crop.w, crop.h);
  const surface = canvas(crop.w * scale, crop.h * scale);
  surface.context.fillStyle = '#808080';
  surface.context.fillRect(0, 0, surface.element.width, surface.element.height);
  surface.context.drawImage(bitmap, crop.x, crop.y, crop.w, crop.h, 0, 0, surface.element.width, surface.element.height);
  return surface.element;
}

/** Stage 1 (many are called): cheap full-range detection over the whole photo and every tile. */
async function proposeFaces(detector: FaceDetector, bitmap: ImageBitmap, onTick: (fraction: number) => void) {
  const crops = detectionCrops(bitmap.width, bitmap.height);
  const proposals: Proposal[] = [];
  for (let index = 0; index < crops.length; index += 1) {
    const crop = crops[index];
    const image = drawCrop(bitmap, crop, Math.min(1024, Math.max(crop.w, crop.h)));
    const scale = crop.w / image.width;
    for (const detection of detector.detect(image).detections) {
      const b = detection.boundingBox;
      if (!b) continue;
      const box = { x: crop.x + b.originX * scale, y: crop.y + b.originY * scale, w: b.width * scale, h: b.height * scale };
      const margin = 0.01 * crop.w;
      // A face cut by an inner tile edge is found whole by an overlapping tile.
      if ((box.x < crop.x + margin && crop.x > 0) || (box.y < crop.y + margin && crop.y > 0)
        || (box.x + box.w > crop.x + crop.w - margin && crop.x + crop.w < bitmap.width - 1)
        || (box.y + box.h > crop.y + crop.h - margin && crop.y + crop.h < bitmap.height - 1)) continue;
      if (box.w < 12) continue;
      proposals.push({ box, score: detection.categories[0]?.score ?? 0 });
    }
    onTick(((index + 1) / crops.length) * 0.6);
    if (index % 4 === 3) await nextFrame();
  }
  proposals.sort((a, b) => b.score - a.score);
  const kept: Proposal[] = [];
  for (const proposal of proposals) if (!kept.some((other) => overlaps(other.box, proposal.box))) kept.push(proposal);
  return kept;
}

/** Stage 2 (few are chosen): the landmarker must confirm each proposal from a close-up crop. */
async function findFaces(bitmap: ImageBitmap, onTick: (fraction: number) => void): Promise<RawFace[]> {
  const { detector, landmarker } = await prepareModels();
  const proposals = await proposeFaces(detector, bitmap, onTick);
  const confirmed: RawFace[] = [];
  for (let index = 0; index < proposals.length; index += 1) {
    const { box } = proposals[index];
    const side = Math.max(box.w, box.h) * 2.4;
    const crop = { x: box.x + box.w / 2 - side / 2, y: box.y + box.h / 2 - side / 2, w: side, h: side };
    const image = drawCrop(bitmap, crop, 384);
    const scale = side / image.width;
    const result = landmarker.detect(image);
    // The crop may include neighbours; take the face nearest its centre.
    let best = -1;
    let bestDistance = Infinity;
    result.faceLandmarks.forEach((landmarks, faceIndex) => {
      const d = Math.hypot(landmarks[1].x - 0.5, landmarks[1].y - 0.5);
      if (d < bestDistance) {
        best = faceIndex;
        bestDistance = d;
      }
    });
    if (best >= 0 && bestDistance < 0.25) {
      const points = result.faceLandmarks[best].slice(0, MESH_SIZE).map((p) => ({ x: crop.x + p.x * image.width * scale, y: crop.y + p.y * image.height * scale }));
      const faceBox = boundsOf(points);
      if (!confirmed.some((other) => overlaps(other.box, faceBox))) {
        confirmed.push({ points, box: faceBox, blend: result.faceBlendshapes[best]?.categories ?? [] });
      }
    }
    onTick(0.6 + ((index + 1) / Math.max(1, proposals.length)) * 0.4);
    if (index % 3 === 2) await nextFrame();
  }
  return confirmed.sort((a, b) => a.box.x - b.box.x);
}

function meanColour(bitmap: ImageBitmap, box: Box): number[] | null {
  const x = Math.max(0, box.x);
  const y = Math.max(0, box.y);
  const w = Math.min(bitmap.width, box.x + box.w) - x;
  const h = Math.min(bitmap.height, box.y + box.h) - y;
  if (w < box.w * 0.5 || h < box.h * 0.5 || w < 2 || h < 2) return null;
  const { context } = canvas(4, 4);
  context.drawImage(bitmap, x, y, w, h, 0, 0, 4, 4);
  const data = context.getImageData(0, 0, 4, 4).data;
  const sum = [0, 0, 0];
  for (let i = 0; i < data.length; i += 4) for (let c = 0; c < 3; c += 1) sum[c] += data[i + c];
  return sum.map((value) => value / 16 / 255);
}

function eyeSharpness(bitmap: ImageBitmap, points: Point[], eyeDistance: number) {
  const eyes = boundsOf([33, 133, 159, 145, 362, 263, 386, 374].map((index) => points[index]));
  const pad = eyeDistance * 0.25;
  const region = { x: eyes.x - pad, y: eyes.y - pad, w: eyes.w + pad * 2, h: eyes.h + pad * 2 };
  const width = 96;
  const height = Math.max(8, Math.round((region.h / region.w) * width));
  const { context } = canvas(width, height);
  context.drawImage(bitmap, region.x, region.y, region.w, region.h, 0, 0, width, height);
  const data = context.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  for (let i = 0; i < gray.length; i += 1) gray[i] = data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114;
  let total = 0, squared = 0, count = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      const laplacian = gray[i - width] + gray[i - 1] - 4 * gray[i] + gray[i + 1] + gray[i + width];
      total += laplacian; squared += laplacian * laplacian; count += 1;
    }
  }
  const variance = squared / Math.max(1, count) - (total / Math.max(1, count)) ** 2;
  return clamp(1 - Math.exp(-Math.max(0, variance) / 120));
}

/** Eye-aligned 24×28 grayscale patch; rows covering the eyes are dropped because blinks change them. */
function alignedTexture(bitmap: ImageBitmap, leftEye: Point, rightEye: Point) {
  const width = 24, height = 28;
  const eyeDistance = Math.max(1, distance(leftEye, rightEye));
  const scale = 10 / eyeDistance;
  const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
  const cos = Math.cos(-angle) * scale, sin = Math.sin(-angle) * scale;
  const { context } = canvas(width, height);
  context.setTransform(cos, sin, -sin, cos, 7 - (cos * leftEye.x - sin * leftEye.y), 9 - (sin * leftEye.x + cos * leftEye.y));
  context.drawImage(bitmap, 0, 0);
  const data = context.getImageData(0, 0, width, height).data;
  const values: number[] = [];
  for (let y = 0; y < height; y += 1) {
    if (y >= 5 && y <= 13) continue;
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      values.push(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
    }
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const norm = Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0)) || 1;
  return Float32Array.from(values, (value) => (value - mean) / norm);
}

function relationSignature(points: Point[]) {
  const left = points[33];
  const right = points[263];
  const eyeDistance = Math.max(1, distance(left, right));
  const axis = { x: (right.x - left.x) / eyeDistance, y: (right.y - left.y) / eyeDistance };
  const center = mid(left, right);
  return [1, 10, 152, 61, 291, 13, 14, 234, 454, 168, 78, 308].flatMap((index) => {
    const dx = points[index].x - center.x;
    const dy = points[index].y - center.y;
    return [(dx * axis.x + dy * axis.y) / eyeDistance, (-dx * axis.y + dy * axis.x) / eyeDistance];
  });
}

function cropThumb(bitmap: ImageBitmap, center: Point, side: number, size: number) {
  const { element, context } = canvas(size, size);
  context.fillStyle = '#dfe3dc';
  context.fillRect(0, 0, size, size);
  context.drawImage(bitmap, center.x - side / 2, center.y - side / 2, side, side, 0, 0, size, size);
  return element.toDataURL('image/jpeg', 0.86);
}

function describeFace(raw: RawFace, bitmap: ImageBitmap, photoId: string): Face {
  const { points, box } = raw;
  const score = (name: string) => raw.blend.find((category) => category.categoryName === name)?.score ?? 0;
  const aspect = (eye: number[]) =>
    (distance(points[eye[1]], points[eye[5]]) + distance(points[eye[2]], points[eye[4]])) / (2 * Math.max(1, distance(points[eye[0]], points[eye[3]])));
  const openness = (blink: number, ear: number) => 0.6 * clamp(1 - (blink - 0.15) / 0.45) + 0.4 * clamp((ear - 0.11) / 0.16);
  const leftOpen = openness(score('eyeBlinkLeft'), aspect(LEFT_EYE));
  const rightOpen = openness(score('eyeBlinkRight'), aspect(RIGHT_EYE));
  const leftEye = mid(points[33], points[133]);
  const rightEye = mid(points[362], points[263]);
  const eyeCenter = mid(points[33], points[263]);
  const eyeDistance = Math.max(1, distance(points[33], points[263]));
  const mouth = mid(points[61], points[291]);
  const center = { x: box.x + box.w / 2, y: box.y + box.h / 2 };
  const up = { x: (points[10].x - points[152].x) / Math.max(1, box.h), y: (points[10].y - points[152].y) / Math.max(1, box.h) };
  const around = (point: Point, along: number, w: number, h: number): Box => ({
    x: point.x + up.x * along * box.h - (w * box.w) / 2,
    y: point.y + up.y * along * box.h - (h * box.h) / 2,
    w: w * box.w,
    h: h * box.h,
  });
  return {
    id: uid('face'),
    photoId,
    points,
    box,
    center,
    size: box.w,
    eyesOpen: 0.5 * Math.min(leftOpen, rightOpen) + 0.25 * (leftOpen + rightOpen),
    smile: (score('mouthSmileLeft') + score('mouthSmileRight')) / 2,
    sharpness: eyeSharpness(bitmap, points, eyeDistance),
    yaw: (points[1].x - eyeCenter.x) / eyeDistance,
    pitch: (points[1].y - eyeCenter.y) / Math.max(1, mouth.y - eyeCenter.y),
    roll: Math.atan2(points[263].y - points[33].y, points[263].x - points[33].x),
    signature: relationSignature(points),
    texture: alignedTexture(bitmap, leftEye, rightEye),
    context: {
      hair: meanColour(bitmap, around(points[10], 0.22, 0.55, 0.22)),
      skin: meanColour(bitmap, { x: points[50].x - box.w * 0.06, y: points[50].y - box.w * 0.06, w: box.w * 0.12, h: box.w * 0.12 })
        ?? meanColour(bitmap, { x: points[280].x - box.w * 0.06, y: points[280].y - box.w * 0.06, w: box.w * 0.12, h: box.w * 0.12 }) ?? [0.5, 0.5, 0.5],
      clothes: meanColour(bitmap, around(points[152], -0.85, 1.2, 0.45)),
    },
    thumb: cropThumb(bitmap, center, Math.max(box.w, box.h) * 1.7, 192),
  };
}

export async function readPhoto(file: File, onTick: (fraction: number) => void): Promise<Photo> {
  const bitmap = await decode(file, WORKING_SIDE);
  const id = uid('photo');
  const raw = await findFaces(bitmap, onTick);
  const thumbScale = 360 / Math.max(bitmap.width, bitmap.height);
  const { element, context } = canvas(bitmap.width * thumbScale, bitmap.height * thumbScale);
  context.drawImage(bitmap, 0, 0, element.width, element.height);
  return {
    id,
    file,
    name: file.name,
    bitmap,
    width: bitmap.width,
    height: bitmap.height,
    thumb: element.toDataURL('image/jpeg', 0.82),
    faces: raw.map((face) => describeFace(face, bitmap, id)),
  };
}
