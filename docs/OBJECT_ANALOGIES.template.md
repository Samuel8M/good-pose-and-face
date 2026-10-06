<!-- Source for docs/OBJECT_ANALOGIES.md. Edit this file, then run: node scripts/build-object-knowledge.mjs -->
# Object analogies: what every detectable thing *is* to the people in a photo

Good Pose & Face can detect 80 kinds of things (the COCO label set, read straight from the detector model's embedded `labels.txt`). This document explains how the app understands each one. The machine-readable version the app loads is [`src/engine/object-knowledge.json`](../src/engine/object-knowledge.json).

## The idea: understand objects by the role they play

Structure-mapping theory (Gentner, 1983) says an analogy is judged by shared **relations**, not shared looks. A bicycle and a horse look nothing alike, but *a person riding a bicycle* and *a person riding a horse* have the same structure: the person is on top of it, straddling it, and moving with it. So for editing a photo, a bicycle is to its rider what a horse is to its rider. A chair is to its sitter what a bench or a couch is to theirs.

To find those relations from evidence rather than guesswork, the app uses **HICO-DET** (Chao et al., 2018), a research benchmark of 600 human–object interactions over exactly these 80 objects ("riding a bicycle", "sitting on a bench", "wearing a tie", "drinking with a cup"…). The 520 real interactions are stored in [`research/hico-det-interactions.json`](../research/hico-det-interactions.json). Grouping their verbs gives a small set of **interaction schemas**: affordances in Gibson's sense, meaning what a thing lets a person do with it.

## Stored as structure-mapping descriptions

Every object is stored the way structure-mapping represents knowledge: entities, attributes, relations, and higher-order relations between relations. A bicycle, for example (`description` in the JSON):

```
(vehicle-family X) (inanimate X) (heavy X) (furniture-sized X) (found-floor X) (rideable X)
(riding person X) (straddling person X) (pushing person X) (walking person X) …
(rides person X)
(cause (rides person X) (supports X person))
```

The last line is the physics of the role: the bicycle holds the person up. A horse has the same causal structure. That is why the Structure-Mapping Engine ([`sme.ts`](../src/engine/sme.ts)) finds *bicycle ≈ horse* although they share almost no attributes. The physical laws behind these structures are explained in [WORLD_MODEL.md](WORLD_MODEL.md).

## Interaction schemas

Each schema is one relational structure between a person and a thing. It also tells the editor what to do with the thing when its person is swapped in from another shot.

{{SCHEMA_TABLE}}

A thing can belong to several schemas. Skis are both *worn* and *ridden*; a dog is a *companion* and can also be *held*. The relation actually seen in the photo decides which one applies. A dog in someone's arms moves with them; a dog at their feet stays.

## How the app uses this

1. **Recognising relations** (`scene.ts`). *Holds* is only considered for things you can grasp, so nobody "holds the dining table". *Carries/wears* applies to worn things. *Sits on* applies to seats. The new **rides** relation covers bicycles, motorcycles, boards, skis and ridden animals.
2. **Checking detections against common sense** (`knowledge.ts`), for better recognition:
   - **Size.** Each object's size is compared with the people standing at the same depth. A "cup" as tall as a person is not a cup. If a look-alike's size fits, the detection is relabelled (cup → vase); otherwise a weak detection is dropped.
   - **Belonging.** Worn things (tie, backpack, baseball glove) must sit on a person. A "tie" floating in the background, or one up at the collar line, is dropped as a false alarm.
3. **Mapping between shots** (`analogy.ts`). A detector may call the same seat a *chair* in one shot and a *bench* in the next. Look-alikes may now map to each other, at a reduced score.
4. **Deciding what else changes** (`memory.ts`, `physics.ts`). Each situation is described structurally and mapped onto physical laws and remembered choices. The objects' schema attributes let a lesson transfer by analogy: teaching "the chair comes along" carries over to couches and benches (same role), but not to cups. See [WORLD_MODEL.md](WORLD_MODEL.md).

## All 80 objects

*Closest analogues* are SME similarities between the objects' structured descriptions (0–1), so shared relational structure outweighs shared looks. *Look-alikes* are things a detector visually confuses with it.

{{OBJECT_TABLE}}

## Limits

- Sizes are deliberately wide common-sense ranges, not measurements. They only catch gross errors (a "cup" the size of a person), and only when people at the same depth give a scale.
- HICO-DET lists what people *can* do with an object, not how often. Some verbs are rare; for example, carrying a chair is real but unusual in group photos. Curated overrides in the build script correct the cases where that would mislead.
- Only the 80 COCO kinds of things are detected. Everything else is background, which the alignment uses but the editor never moves.

## Sources

- Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014: labels and super-categories.
- Chao, Liu, Liu, Zeng & Deng, *Learning to Detect Human-Object Interactions*, WACV 2018: HICO-DET interaction categories (copied from the open-source [GEN-VLKT](https://github.com/YueLiao/gen-vlkt) code).
- Gentner, *Structure-mapping: A theoretical framework for analogy*, Cognitive Science 7, 1983.
- Forbus, Gentner & Law, *MAC/FAC: A model of similarity-based retrieval*, Cognitive Science 19, 1995.
- Gibson, *The Ecological Approach to Visual Perception*, 1979: affordances.
