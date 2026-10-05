# Good Pose & Face

Swap in anyone's best **face**, best **pose**, or a better version of anything in a group photo, using a few shots taken at the same moment. The app understands the scene, so it knows what else has to change with your edit and what must stay untouched.

**Use it:** https://samuel8m.github.io/good-pose-and-face/

Everything runs in the browser. Photos are never uploaded.

## How to use it

1. Add 2 or more photos of the same moment (button or drag and drop).
2. Tap a face ring or a label in the picture, such as *Chair* or *Dog*.
3. For a person, choose **Better face** (just the face) or **Better pose** (the whole person from another shot).
4. Check **What else changes**. Each related thing shows whether it comes along or stays, and why. Tap to change any decision; the app learns from that.
5. **Save photo**. The saved photo is rebuilt from the originals at full resolution.

## The analogy engine

The design follows structure-mapping theory (Gentner) and its computational models: SME (Falkenhainer, Forbus & Gentner) and MAC/FAC retrieval (Forbus, Gentner & Law).

| Stage | File | What it does |
|---|---|---|
| Faces | `src/engine/vision.ts` | Two-stage face finding: a full-range detector over tiles proposes faces, then the 468-point landmarker confirms each one. Eye openness, pose and sharpness are measured per face. |
| Scene | `src/engine/scene.ts` | Turns each photo into a relational description. Entities are people (body pose and linked face) and objects (COCO classes), each with a precise outline from MagicTouch interactive segmentation. Relations are *holds*, *carries*, *sits on*, *in front of*, *touches*, *left of* and *above*. |
| Analogy | `src/engine/analogy.ts` | Aligns the shots on their background, then maps entities. A cup only maps to a cup (identicality). Local matches are rescored by systematicity, meaning how many of their relations the rest of the mapping preserves. |
| Edit planning | `src/engine/analogy.ts`, `src/engine/cases.ts` | For a swap, each related entity is described by its relation to the swapped one. A case library judges it by analogy: MAC filtering, then FAC structural similarity, then a weighted vote of the nearest cases. Built-in cases hold common sense (held things move, seats stay, people in front stay in front). |
| Learning | `src/engine/cases.ts` | Each override is stored as a new, heavier case in `localStorage` and generalizes to analogous situations. For example, teaching "the chair comes along" also changes the verdict for a bench being sat on, but not for a cup or a neighbour. |
| Compositing | `src/engine/composeScene.ts`, `src/engine/compose.ts` | A whole-person swap takes the aligned donor shot inside the union of the old and new outlines plus everything that moves, minus everything that stays. Occluders always stay on top; things behind give way where the new outline is. Exposure is matched on the seam. Face swaps use a landmark mesh warp that keeps the donor's expression, a lighting plane fit, and detail restoration. |

Faces narrower than 60 px are never labelled open or closed. Matching is a same-session heuristic, not identity verification. Please only edit photos of people who'd be happy with it, and keep the originals.

## Develop

```sh
npm install
npm run dev          # http://localhost:5185
npm run typecheck
npm run build:pages  # static build with relative paths, for GitHub Pages
```

The MediaPipe WASM runtime is bundled from the installed `@mediapipe/tasks-vision` (pinned to 0.10.35), so the runtime always matches the JS API. The models download from Google's MediaPipe model storage on first use. The pose landmarker's own segmentation masks are disabled because they abort on the CPU delegate in 0.10.35; outlines come from the interactive segmenter instead.

This project grew out of [Good Take](https://github.com/Samuel8M/good-take) (faces only).
