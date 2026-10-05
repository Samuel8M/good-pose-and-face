import { Delaunay } from 'd3-delaunay';
import { drawSceneSwap, type EditRegion, type SceneSwap } from './composeScene';
import { decodeFull, distance, OVAL, type Box, type Face, type Photo, type Point } from './vision';

export type Swap = { target: Face; donor: Face; donorPhoto: Photo };
export type PlacedSceneSwap = { swap: SceneSwap; region: EditRegion };
type Pixels = Pick<Photo, 'bitmap' | 'width' | 'height'>;

// iOS Safari refuses canvases over 16.7 megapixels; elsewhere the limit is far higher.
const IS_IOS = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.userAgent.includes('Macintosh') && navigator.maxTouchPoints > 1);
const MAX_CANVAS_AREA = IS_IOS ? 16_777_216 : 120_000_000;

// Landmarks that move with the head but barely with expression: eye corners, nose, forehead, chin, cheeks.
const STABLE = [33, 133, 362, 263, 1, 4, 6, 168, 197, 195, 10, 152, 234, 454];

function surface(width: number, height: number) {
  const element = document.createElement('canvas');
  element.width = Math.max(1, Math.round(width));
  element.height = Math.max(1, Math.round(height));
  const context = element.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('This browser could not open a drawing surface.');
  context.imageSmoothingQuality = 'high';
  return { element, context };
}

function regionAround(box: Box, pad: number, width: number, height: number): Box {
  const x = Math.max(0, Math.floor(box.x - box.w * pad));
  const y = Math.max(0, Math.floor(box.y - box.h * pad));
  return { x, y, w: Math.min(width, Math.ceil(box.x + box.w * (1 + pad))) - x, h: Math.min(height, Math.ceil(box.y + box.h * (1 + pad))) - y };
}

/** Least-squares similarity transform (rotation, uniform scale, shift) taking `from` onto `to`. */
function similarityFit(from: Point[], to: Point[]) {
  const n = from.length;
  const fromMean = { x: from.reduce((s, p) => s + p.x, 0) / n, y: from.reduce((s, p) => s + p.y, 0) / n };
  const toMean = { x: to.reduce((s, p) => s + p.x, 0) / n, y: to.reduce((s, p) => s + p.y, 0) / n };
  let a = 0, b = 0, norm = 0;
  for (let i = 0; i < n; i += 1) {
    const fx = from[i].x - fromMean.x, fy = from[i].y - fromMean.y;
    const tx = to[i].x - toMean.x, ty = to[i].y - toMean.y;
    a += fx * tx + fy * ty;
    b += fx * ty - fy * tx;
    norm += fx * fx + fy * fy;
  }
  const cos = a / Math.max(1e-9, norm), sin = b / Math.max(1e-9, norm);
  return (p: Point) => ({
    x: cos * (p.x - fromMean.x) - sin * (p.y - fromMean.y) + toMean.x,
    y: sin * (p.x - fromMean.x) + cos * (p.y - fromMean.y) + toMean.y,
  });
}

/**
 * Where each donor landmark should land. The donor's expression (open eyes, smile) is kept by
 * moving its whole mesh rigidly onto the target head; only the outline is bent to the target's
 * jaw and forehead, with the bend fading out toward the eyes and mouth.
 */
function destinationMesh(target: Face, donor: Face) {
  const fit = similarityFit(STABLE.map((i) => donor.points[i]), STABLE.map((i) => target.points[i]));
  const placed = donor.points.map(fit);
  const residuals = OVAL.map((i) => ({ at: placed[i], dx: target.points[i].x - placed[i].x, dy: target.points[i].y - placed[i].y }));
  const falloff = 0.16 * target.size;
  return placed.map((point) => {
    let wx = 0, wy = 0, wsum = 0, nearest = Infinity;
    for (const r of residuals) {
      const d = distance(point, r.at);
      nearest = Math.min(nearest, d);
      const w = 1 / (d * d + 1);
      wx += w * r.dx; wy += w * r.dy; wsum += w;
    }
    const bend = Math.exp(-((nearest / falloff) ** 2));
    return { x: point.x + (bend * wx) / wsum, y: point.y + (bend * wy) / wsum };
  });
}

function affine(from: Point[], to: Point[]) {
  const [s0, s1, s2] = from;
  const [t0, t1, t2] = to;
  const dx1 = s1.x - s0.x, dy1 = s1.y - s0.y, dx2 = s2.x - s0.x, dy2 = s2.y - s0.y;
  const det = dx1 * dy2 - dx2 * dy1;
  if (Math.abs(det) < 1e-6) return null;
  const a = ((t1.x - t0.x) * dy2 - (t2.x - t0.x) * dy1) / det;
  const c = (dx1 * (t2.x - t0.x) - dx2 * (t1.x - t0.x)) / det;
  const b = ((t1.y - t0.y) * dy2 - (t2.y - t0.y) * dy1) / det;
  const d = (dx1 * (t2.y - t0.y) - dx2 * (t1.y - t0.y)) / det;
  return [a, b, c, d, t0.x - a * s0.x - c * s0.y, t0.y - b * s0.x - d * s0.y] as const;
}

/** Separable box blur run three times (≈ Gaussian) on a single-channel mask. */
function blurMask(mask: Float32Array, width: number, height: number, radius: number) {
  const temp = new Float32Array(mask.length);
  const pass = (src: Float32Array, dst: Float32Array, horizontal: boolean) => {
    const lines = horizontal ? height : width;
    const length = horizontal ? width : height;
    const at = (line: number, k: number) => (horizontal ? line * width + k : k * width + line);
    for (let line = 0; line < lines; line += 1) {
      let sum = 0;
      for (let k = -radius; k <= radius; k += 1) sum += src[at(line, Math.min(length - 1, Math.max(0, k)))];
      for (let k = 0; k < length; k += 1) {
        dst[at(line, k)] = sum / (2 * radius + 1);
        sum += src[at(line, Math.min(length - 1, k + radius + 1))] - src[at(line, Math.max(0, k - radius))];
      }
    }
  };
  for (let i = 0; i < 3; i += 1) {
    pass(mask, temp, true);
    pass(temp, mask, false);
  }
}

/** Fit a brightness plane (a + b·u + c·v) per channel to target − donor, trimming outliers like eyes that changed. */
function lightingPlanes(donor: Uint8ClampedArray, target: Uint8ClampedArray, weight: Float32Array, width: number, height: number) {
  const planes: number[][] = [];
  for (let channel = 0; channel < 3; channel += 1) {
    let plane = [0, 0, 0];
    for (let round = 0; round < 2; round += 1) {
      const m = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      const r = [0, 0, 0];
      let count = 0;
      for (let y = 0; y < height; y += 2) {
        for (let x = 0; x < width; x += 2) {
          const i = y * width + x;
          if (weight[i] < 0.7 || donor[i * 4 + 3] < 250) continue;
          const u = (2 * x) / width - 1, v = (2 * y) / height - 1;
          const diff = target[i * 4 + channel] - donor[i * 4 + channel];
          if (round === 1 && Math.abs(diff - (plane[0] + plane[1] * u + plane[2] * v)) > 28) continue;
          const basis = [1, u, v];
          for (let p = 0; p < 3; p += 1) {
            r[p] += basis[p] * diff;
            for (let q = 0; q < 3; q += 1) m[p * 3 + q] += basis[p] * basis[q];
          }
          count += 1;
        }
      }
      if (count < 30) break;
      plane = solve3(m, r) ?? [r[0] / count, 0, 0];
    }
    planes.push(plane);
  }
  return planes;
}

function solve3(m: number[], r: number[]) {
  const det = (a: number[]) => a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  const d = det(m);
  if (Math.abs(d) < 1e-9) return null;
  return [0, 1, 2].map((column) => {
    const copy = [...m];
    for (let row = 0; row < 3; row += 1) copy[row * 3 + column] = r[row];
    return det(copy) / d;
  });
}

/** Fine-detail energy (variance of the luminance Laplacian) inside the given rectangle. */
function detail(data: Uint8ClampedArray, width: number, rect: Box) {
  const lum = (i: number) => data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114;
  let sum = 0, squared = 0, count = 0;
  for (let y = Math.max(1, Math.round(rect.y)); y < Math.round(rect.y + rect.h) - 1; y += 1) {
    for (let x = Math.max(1, Math.round(rect.x)); x < Math.min(width - 1, Math.round(rect.x + rect.w)); x += 1) {
      const i = y * width + x;
      const value = lum(i - width) + lum(i - 1) - 4 * lum(i) + lum(i + 1) + lum(i + width);
      sum += value; squared += value * value; count += 1;
    }
  }
  return count ? squared / count - (sum / count) ** 2 : 0;
}

/**
 * Warping resamples every pixel between its neighbours, which softens fine detail (measured at
 * roughly half lost). Unsharp-mask the patch just enough to bring its detail back to the wanted level.
 */
function restoreDetail(patch: ImageData, core: Box, wanted: number) {
  const { data, width, height } = patch;
  if (detail(data, width, core) >= wanted * 0.95) return;
  const blurred = new Float32Array(data.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      for (let c = 0; c < 3; c += 1) {
        let total = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            const sx = Math.min(width - 1, Math.max(0, x + dx)), sy = Math.min(height - 1, Math.max(0, y + dy));
            total += data[(sy * width + sx) * 4 + c] * (dx ? 1 : 2) * (dy ? 1 : 2);
          }
        }
        blurred[(y * width + x) * 4 + c] = total / 16;
      }
    }
  }
  const sharpened = (amount: number) => {
    const out = new Uint8ClampedArray(data);
    for (let i = 0; i < out.length; i += 1) if (i % 4 !== 3) out[i] = data[i] + amount * (data[i] - blurred[i]);
    return out;
  };
  let low = 0, high = 2.5;
  for (let step = 0; step < 7; step += 1) {
    const amount = (low + high) / 2;
    if (detail(sharpened(amount), width, core) < wanted) low = amount; else high = amount;
  }
  data.set(sharpened(low));
}

/** Renders one replacement face as a transparent patch positioned over the target photo. */
function swapPatch(base: Pixels, target: Face, donor: Face, donorPhoto: Pixels) {
  const region = regionAround(target.box, 0.12, base.width, base.height);
  const donorRegion = regionAround(donor.box, 0.25, donorPhoto.width, donorPhoto.height);
  const donorCrop = surface(donorRegion.w, donorRegion.h);
  donorCrop.context.drawImage(donorPhoto.bitmap, donorRegion.x, donorRegion.y, donorRegion.w, donorRegion.h, 0, 0, donorRegion.w, donorRegion.h);

  const source = donor.points.map((p) => ({ x: p.x - donorRegion.x, y: p.y - donorRegion.y }));
  const dest = destinationMesh(target, donor).map((p) => ({ x: p.x - region.x, y: p.y - region.y }));

  const warped = surface(region.w, region.h);
  const triangles = Delaunay.from(dest, (p) => p.x, (p) => p.y).triangles;
  for (let i = 0; i < triangles.length; i += 3) {
    const ids = [triangles[i], triangles[i + 1], triangles[i + 2]];
    const to = ids.map((id) => dest[id]);
    const transform = affine(ids.map((id) => source[id]), to);
    if (!transform) continue;
    // Grow each triangle a hair so anti-aliased edges don't leave seams between neighbours.
    const cx = (to[0].x + to[1].x + to[2].x) / 3, cy = (to[0].y + to[1].y + to[2].y) / 3;
    const ctx = warped.context;
    ctx.save();
    ctx.beginPath();
    to.forEach((p, k) => {
      const len = Math.hypot(p.x - cx, p.y - cy) || 1;
      const x = p.x + ((p.x - cx) / len) * 0.7, y = p.y + ((p.y - cy) / len) * 0.7;
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(...transform);
    ctx.drawImage(donorCrop.element, 0, 0);
    ctx.restore();
  }

  // Soft mask: the target outline pulled slightly inward, then feathered, so the seam sits on skin.
  const maskSurface = surface(region.w, region.h);
  const ovalCenter = OVAL.reduce((c, i) => ({ x: c.x + dest[i].x / OVAL.length, y: c.y + dest[i].y / OVAL.length }), { x: 0, y: 0 });
  maskSurface.context.fillStyle = '#fff';
  maskSurface.context.beginPath();
  OVAL.forEach((i, k) => {
    const x = ovalCenter.x + (dest[i].x - ovalCenter.x) * 0.9;
    const y = ovalCenter.y + (dest[i].y - ovalCenter.y) * 0.9;
    if (k === 0) maskSurface.context.moveTo(x, y); else maskSurface.context.lineTo(x, y);
  });
  maskSurface.context.fill();
  const maskPixels = maskSurface.context.getImageData(0, 0, region.w, region.h).data;
  const mask = new Float32Array(region.w * region.h);
  for (let i = 0; i < mask.length; i += 1) mask[i] = maskPixels[i * 4 + 3] / 255;
  blurMask(mask, region.w, region.h, Math.max(2, Math.round(target.size * 0.035)));

  const original = surface(region.w, region.h);
  original.context.drawImage(base.bitmap, region.x, region.y, region.w, region.h, 0, 0, region.w, region.h);
  const targetData = original.context.getImageData(0, 0, region.w, region.h).data;
  const patch = warped.context.getImageData(0, 0, region.w, region.h);
  // Compare the middle of the face (eyes, nose, mouth) before and after warping.
  const middle = (box: Box, origin: Box): Box => ({ x: box.x - origin.x + box.w * 0.25, y: box.y - origin.y + box.h * 0.25, w: box.w * 0.5, h: box.h * 0.5 });
  const donorDetail = detail(donorCrop.context.getImageData(0, 0, donorRegion.w, donorRegion.h).data, donorRegion.w, middle(donor.box, donorRegion));
  // Enlarging a smaller face can't invent detail, so expect proportionally less.
  const enlargement = Math.max(1, target.size / donor.size);
  restoreDetail(patch, middle(target.box, region), donorDetail / enlargement ** 2);
  const planes = lightingPlanes(patch.data, targetData, mask, region.w, region.h);
  for (let y = 0; y < region.h; y += 1) {
    for (let x = 0; x < region.w; x += 1) {
      const i = y * region.w + x;
      const o = i * 4;
      if (patch.data[o + 3] === 0 || mask[i] <= 0.002) {
        patch.data[o + 3] = 0;
        continue;
      }
      const u = (2 * x) / region.w - 1, v = (2 * y) / region.h - 1;
      for (let c = 0; c < 3; c += 1) patch.data[o + c] += planes[c][0] + planes[c][1] * u + planes[c][2] * v;
      patch.data[o + 3] = Math.round(mask[i] * patch.data[o + 3]);
    }
  }
  warped.context.putImageData(patch, 0, 0);
  return { element: warped.element, x: region.x, y: region.y };
}

/** Draws every chosen replacement onto a transparent overlay the size of the base photo. */
/** Whole-person/object swaps first (they replace large areas), then face swaps on top. */
export function renderOverlay(overlay: HTMLCanvasElement, base: Photo, swaps: Swap[], sceneSwaps: PlacedSceneSwap[] = []) {
  overlay.width = base.width;
  overlay.height = base.height;
  const context = overlay.getContext('2d');
  if (!context) return;
  context.clearRect(0, 0, overlay.width, overlay.height);
  for (const { swap, region } of sceneSwaps) drawSceneSwap(context, base, 1, swap.donorPhoto, 1, swap, region);
  for (const swap of swaps) {
    const patch = swapPatch(base, swap.target, swap.donor, swap.donorPhoto);
    context.drawImage(patch.element, patch.x, patch.y);
  }
}

function scaleFace(face: Face, k: number): Face {
  if (k === 1) return face;
  return {
    ...face,
    points: face.points.map((p) => ({ x: p.x * k, y: p.y * k })),
    box: { x: face.box.x * k, y: face.box.y * k, w: face.box.w * k, h: face.box.h * k },
    center: { x: face.center.x * k, y: face.center.y * k },
    size: face.size * k,
  };
}

/**
 * Rebuilds the edit from the original files at full resolution (the on-screen copy may be
 * smaller) and saves at JPEG quality 1, so untouched areas stay as crisp as the originals.
 */
export async function exportJpeg(main: Photo, swaps: Swap[], sceneSwaps: PlacedSceneSwap[] = []) {
  const opened = new Map<string, Pixels & { k: number }>();
  const open = async (photo: Photo) => {
    let entry = opened.get(photo.id);
    if (!entry) {
      let bitmap = await decodeFull(photo.file);
      const fit = Math.min(1, Math.sqrt(MAX_CANVAS_AREA / (bitmap.width * bitmap.height)));
      if (fit < 1) {
        const resized = await createImageBitmap(bitmap, { resizeWidth: Math.floor(bitmap.width * fit), resizeHeight: Math.floor(bitmap.height * fit), resizeQuality: 'high' });
        bitmap.close();
        bitmap = resized;
      }
      entry = { bitmap, width: bitmap.width, height: bitmap.height, k: bitmap.width / photo.width };
      opened.set(photo.id, entry);
    }
    return entry;
  };
  try {
    const base = await open(main);
    const { element, context } = surface(base.width, base.height);
    context.drawImage(base.bitmap, 0, 0);
    for (const { swap, region } of sceneSwaps) {
      const donor = await open(swap.donorPhoto);
      drawSceneSwap(context, base, base.k, donor, donor.k, swap, region);
    }
    for (const swap of swaps) {
      const donor = await open(swap.donorPhoto);
      const patch = swapPatch(base, scaleFace(swap.target, base.k), scaleFace(swap.donor, donor.k), donor);
      context.drawImage(patch.element, patch.x, patch.y);
    }
    return await new Promise<Blob>((resolve, reject) => {
      element.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('The photo could not be saved.'))), 'image/jpeg', 1);
    });
  } finally {
    opened.forEach((entry) => entry.bitmap.close());
  }
}
