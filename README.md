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
| Scene | `src/engine/scene.ts` | Turns each photo into a relational description. Entities are people (body pose and linked face) and objects (COCO classes), each with a precise outline from MagicTouch interactive segmentation. Relations are *holds*, *carries*, *sits on*, *rides*, *in front of*, *touches*, *left of* and *above*. |
| Analogy | `src/engine/analogy.ts` | Aligns the shots on their background, then maps entities. A cup only maps to a cup (identicality). Local matches are rescored by systematicity, meaning how many of their relations the rest of the mapping preserves. |
| Object knowledge | `src/engine/knowledge.ts`, `src/engine/object-knowledge.json` | All 80 detectable things, stored as structure-mapping descriptions built from 520 real human–object interactions (HICO-DET). Each has interaction schemas (grasp, wear, ride, seat…), typical size, look-alikes, and analogues computed by SME. Detections are sanity-checked against people at the same depth: wrong-sized ones are relabelled or dropped. See [docs/OBJECT_ANALOGIES.md](docs/OBJECT_ANALOGIES.md). |
| Structure-mapping | `src/engine/sme.ts` | A compact Structure-Mapping Engine: local matches, structurally consistent kernels, greedy one-to-one merge, systematicity scoring and candidate inferences. |
| Physics | `src/engine/physics.ts` | Physical laws as base domains (gravity and support, rigid attachment, rider systems, seats on the ground, occlusion, inertia, object permanence…) that photo situations are mapped onto. Lambertian lighting with a first-order spherical-harmonic fit relights swapped faces to the target photo's light. See [docs/WORLD_MODEL.md](docs/WORLD_MODEL.md). |
| Memory and dreaming | `src/engine/memory.ts` | Complementary-learning-systems design: choices are stored at once as structured episodes. While idle the app "dreams": it merges analogous episodes into probabilistic generalizations (SAGE-style), maps them onto the physical laws, and extends them to analogous objects. Insights appear in the app. |
| Compositing | `src/engine/composeScene.ts`, `src/engine/compose.ts` | A whole-person swap takes the aligned donor shot inside the union of the old and new outlines plus everything that moves, minus everything that stays. Occluders always stay on top; things behind give way where the new outline is. Exposure is matched on the seam. Face swaps use a landmark mesh warp that keeps the donor's expression, Lambertian relighting, a lighting plane fit, and detail restoration. |

Faces narrower than 60 px are never labelled open or closed. Matching is a same-session heuristic, not identity verification. Please only edit photos of people who'd be happy with it, and keep the originals.

## Develop

```sh
npm install
npm run dev          # http://localhost:5185
npm run typecheck
npm run build:pages  # static build with relative paths, for GitHub Pages
node scripts/build-object-knowledge.mjs  # regenerate object knowledge + docs (Node 23.6+)
```

The MediaPipe WASM runtime is bundled from the installed `@mediapipe/tasks-vision` (pinned to 0.10.35), so the runtime always matches the JS API. The models download from Google's MediaPipe model storage on first use. The pose landmarker's own segmentation masks are disabled because they abort on the CPU delegate in 0.10.35; outlines come from the interactive segmenter instead.

This project grew out of [Good Take](https://github.com/Samuel8M/good-take) (faces only).
