import type { Involvement, Shift } from './analogy';
import type { Entity, Mask } from './scene';
import { clamp, type Photo } from './vision';

/*
 * Swapping a whole person (or object) from another shot of the same moment.
 *
 * Because the two shots are aligned, the donor shot already contains the right background
 * wherever the old pose used to be. So the edit is: take the aligned donor shot inside a region,
 * keep the main shot everywhere else. The region is the union of the old and new outlines of
 * the swapped thing and everything decided to move with it, minus everything decided to stay
 * put (so a person standing in front, or the chair being sat on, is never painted over).
 */

export type SceneSwap = {
  target: Entity;
  donor: Entity;
  donorPhoto: Photo;
  shift: Shift;
  involved: Involvement[];
};
type Pixels = Pick<Photo, 'bitmap' | 'width' | 'height'>;
export type EditRegion = { alpha: Float32Array; protect: Float32Array; width: number; height: number; cell: number };

const GRID = 900;

/** Paints a mask into main-photo grid cells; `toMask` maps a main-photo point into the mask's photo. */
function paint(target: Float32Array, width: number, height: number, cell: number, mask: Mask, toMask: (x: number, y: number) => { x: number; y: number }) {
  for (let gy = 0; gy < height; gy += 1) {
    for (let gx = 0; gx < width; gx += 1) {
      const p = toMask((gx + 0.5) * cell, (gy + 0.5) * cell);
      const mx = Math.floor(((p.x - mask.box.x) / mask.box.w) * mask.width);
      const my = Math.floor(((p.y - mask.box.y) / mask.box.h) * mask.height);
      if (mx < 0 || my < 0 || mx >= mask.width || my >= mask.height) continue;
      const i = gy * width + gx;
      target[i] = Math.max(target[i], mask.data[my * mask.width + mx]);
    }
  }
}

function grow(grid: Float32Array, width: number, height: number, radius: number) {
  if (radius < 1) return grid;
  const pass = (src: Float32Array, horizontal: boolean) => {
    const out = new Float32Array(src.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let best = 0;
        for (let k = -radius; k <= radius; k += 1) {
          const sx = horizontal ? x + k : x, sy = horizontal ? y : y + k;
          if (sx >= 0 && sy >= 0 && sx < width && sy < height) best = Math.max(best, src[sy * width + sx]);
        }
        out[y * width + x] = best;
      }
    }
    return out;
  };
  return pass(pass(grid, true), false);
}

function soften(grid: Float32Array, width: number, height: number, radius: number) {
  const pass = (src: Float32Array, horizontal: boolean) => {
    const out = new Float32Array(src.length);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = 0, count = 0;
        for (let k = -radius; k <= radius; k += 1) {
          const sx = horizontal ? x + k : x, sy = horizontal ? y : y + k;
          if (sx >= 0 && sy >= 0 && sx < width && sy < height) { sum += src[sy * width + sx]; count += 1; }
        }
        out[y * width + x] = sum / count;
      }
    }
    return out;
  };
  return pass(pass(grid, true), false);
}

/** Where the donor shot replaces the main shot (alpha) and what is shielded from it (protect). */
export function editRegion(main: Photo, swap: SceneSwap): EditRegion {
  const cell = Math.max(main.width, main.height) / GRID;
  const width = Math.ceil(main.width / cell), height = Math.ceil(main.height / cell);
  const take = new Float32Array(width * height);
  // Things that stay: occluders keep covering the swapped thing; things behind it give way
  // wherever the incoming outline is.
  const covers = new Float32Array(width * height);
  const behind = new Float32Array(width * height);
  const incoming = new Float32Array(width * height);
  const same = (x: number, y: number) => ({ x, y });
  const toDonor = (x: number, y: number) => ({ x: x * swap.shift.scale + swap.shift.dx, y: y * swap.shift.scale + swap.shift.dy });

  paint(take, width, height, cell, swap.target.mask, same);
  paint(take, width, height, cell, swap.donor.mask, toDonor);
  paint(incoming, width, height, cell, swap.donor.mask, toDonor);
  for (const item of swap.involved) {
    if (item.decision === 'moves') {
      if (item.entity.photoId === main.id) paint(take, width, height, cell, item.entity.mask, same);
      if (item.counterpart) {
        paint(take, width, height, cell, item.counterpart.mask, toDonor);
        paint(incoming, width, height, cell, item.counterpart.mask, toDonor);
      }
    } else if (item.entity.photoId === main.id) {
      paint(item.inFront ? covers : behind, width, height, cell, item.entity.mask, same);
    }
  }
  // Grow the take region past soft outline edges (hair, motion blur), but keep shielded things whole.
  const reach = Math.max(2, Math.round((swap.target.size / cell) * 0.04));
  const grown = grow(take, width, height, reach);
  const front = grow(covers.map((v) => (v > 0.5 ? 1 : 0)), width, height, 1);
  const back = grow(behind.map((v) => (v > 0.5 ? 1 : 0)), width, height, 1);
  const shield = new Float32Array(width * height);
  const alpha = new Float32Array(width * height);
  for (let i = 0; i < alpha.length; i += 1) {
    shield[i] = Math.max(front[i], back[i] * (1 - incoming[i]));
    alpha[i] = clamp(grown[i] - shield[i]);
  }
  return { alpha: soften(alpha, width, height, Math.max(1, Math.round(reach / 2))), protect: shield, width, height, cell };
}

/**
 * Draws the swap into `context`, whose pixels are the main photo scaled by `k` (1 for the
 * on-screen copy, larger for a full-resolution save). `donor` holds the donor photo scaled by `kd`.
 */
export function drawSceneSwap(context: CanvasRenderingContext2D, base: Pixels, k: number, donor: Pixels, kd: number, swap: SceneSwap, region: EditRegion) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let gy = 0; gy < region.height; gy += 1) {
    for (let gx = 0; gx < region.width; gx += 1) {
      if (region.alpha[gy * region.width + gx] < 0.004) continue;
      x0 = Math.min(x0, gx); x1 = Math.max(x1, gx); y0 = Math.min(y0, gy); y1 = Math.max(y1, gy);
    }
  }
  if (x1 < x0) return;
  const unit = region.cell * k;
  const box = {
    x: Math.max(0, Math.floor(x0 * unit)),
    y: Math.max(0, Math.floor(y0 * unit)),
    w: 0,
    h: 0,
  };
  box.w = Math.min(base.width, Math.ceil((x1 + 1) * unit)) - box.x;
  box.h = Math.min(base.height, Math.ceil((y1 + 1) * unit)) - box.y;

  // Donor shot moved into the main shot's frame: main = (donorPx/kd - d) / s, scaled by k.
  const layer = document.createElement('canvas');
  layer.width = box.w;
  layer.height = box.h;
  const lc = layer.getContext('2d', { willReadFrequently: true })!;
  lc.imageSmoothingQuality = 'high';
  const a = k / (kd * swap.shift.scale);
  lc.setTransform(a, 0, 0, a, (-k * swap.shift.dx) / swap.shift.scale - box.x, (-k * swap.shift.dy) / swap.shift.scale - box.y);
  lc.drawImage(donor.bitmap, 0, 0);
  lc.setTransform(1, 0, 0, 1, 0, 0);

  // Match exposure using the soft rim, where both shots show the same background.
  const mine = document.createElement('canvas');
  mine.width = box.w;
  mine.height = box.h;
  const mc = mine.getContext('2d', { willReadFrequently: true })!;
  mc.drawImage(base.bitmap, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
  const theirs = lc.getImageData(0, 0, box.w, box.h);
  const ours = mc.getImageData(0, 0, box.w, box.h).data;
  const sumsT = [0, 0, 0], sumsO = [0, 0, 0];
  let rim = 0;
  const alphaAt = (px: number, py: number) => {
    const gx = Math.min(region.width - 1, Math.floor((box.x + px) / unit)), gy = Math.min(region.height - 1, Math.floor((box.y + py) / unit));
    return region.alpha[gy * region.width + gx];
  };
  const step = Math.max(1, Math.round(unit / 2));
  for (let py = 0; py < box.h; py += step) {
    for (let px = 0; px < box.w; px += step) {
      const v = alphaAt(px, py);
      if (v < 0.05 || v > 0.6 || theirs.data[(py * box.w + px) * 4 + 3] < 255) continue;
      const i = (py * box.w + px) * 4;
      for (let c = 0; c < 3; c += 1) { sumsT[c] += theirs.data[i + c]; sumsO[c] += ours[i + c]; }
      rim += 1;
    }
  }
  const gain = sumsT.map((t, c) => (rim > 20 ? clamp(sumsO[c] / Math.max(1, t), 0.8, 1.25) : 1));
  // Apply gain and the soft region as transparency, sampled bilinearly from the grid.
  for (let py = 0; py < box.h; py += 1) {
    for (let px = 0; px < box.w; px += 1) {
      const fx = (box.x + px) / unit - 0.5, fy = (box.y + py) / unit - 0.5;
      const gx = Math.max(0, Math.min(region.width - 1, Math.floor(fx))), gy = Math.max(0, Math.min(region.height - 1, Math.floor(fy)));
      const gx1 = Math.min(region.width - 1, gx + 1), gy1 = Math.min(region.height - 1, gy + 1);
      const tx = clamp(fx - gx), ty = clamp(fy - gy);
      const at = (x: number, y: number) => region.alpha[y * region.width + x];
      const v = (at(gx, gy) * (1 - tx) + at(gx1, gy) * tx) * (1 - ty) + (at(gx, gy1) * (1 - tx) + at(gx1, gy1) * tx) * ty;
      const i = (py * box.w + px) * 4;
      for (let c = 0; c < 3; c += 1) theirs.data[i + c] = theirs.data[i + c] * gain[c];
      theirs.data[i + 3] = Math.round(theirs.data[i + 3] * v);
    }
  }
  lc.putImageData(theirs, 0, 0);
  context.drawImage(layer, box.x, box.y);
}

/** Tints the edit region green and shielded things blue, for showing what will change. */
export function drawRegionPreview(canvas: HTMLCanvasElement, main: Photo, regions: EditRegion[]) {
  canvas.width = Math.round(main.width / 4);
  canvas.height = Math.round(main.height / 4);
  const context = canvas.getContext('2d')!;
  const image = context.createImageData(canvas.width, canvas.height);
  for (const region of regions) {
    for (let y = 0; y < canvas.height; y += 1) {
      for (let x = 0; x < canvas.width; x += 1) {
        const gx = Math.min(region.width - 1, Math.floor((x * 4) / region.cell)), gy = Math.min(region.height - 1, Math.floor((y * 4) / region.cell));
        const take = region.alpha[gy * region.width + gx], shield = region.protect[gy * region.width + gx];
        const i = (y * canvas.width + x) * 4;
        if (take > 0.1) { image.data[i] = 40; image.data[i + 1] = 200; image.data[i + 2] = 110; image.data[i + 3] = Math.max(image.data[i + 3], Math.round(110 * take)); }
        else if (shield > 0.5) { image.data[i] = 70; image.data[i + 1] = 130; image.data[i + 2] = 255; image.data[i + 3] = Math.max(image.data[i + 3], 95); }
      }
    }
  }
  context.putImageData(image, 0, 0);
}
