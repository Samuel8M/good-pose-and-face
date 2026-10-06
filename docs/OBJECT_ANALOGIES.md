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

| Schema | When its person is swapped | Members | Why |
|---|---|---|---|
| **grasp** | Moves with the person | bird, cat, dog, backpack, umbrella, handbag, tie, suitcase, frisbee, sports ball, baseball bat, baseball glove, skateboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, laptop, mouse, remote, keyboard, cell phone, book, vase, scissors, teddy bear, hair drier, toothbrush | It is in their hands, so it moves with them. |
| **wear** | Moves with the person | backpack, handbag, tie, skis, snowboard, baseball glove | They are wearing it, so it moves with them. |
| **ride** | Moves with the person | bicycle, motorcycle, horse, elephant, skis, snowboard, skateboard, surfboard | Rider and ride move together between shots. |
| **seat** | Stays in place | bench, chair, couch, bed, toilet | Seats and beds stay where they are; the person sits on them. |
| **table** | Stays in place | dining table | Tables stay put. |
| **enclose** | Stays in place | car, airplane, bus, train, truck, boat | Vehicles they are in or beside stay where they are. |
| **companion** | Stays in place | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe | Animals have their own pose; swap them separately unless they are being carried. |
| **fixture** | Stays in place | traffic light, fire hydrant, stop sign, parking meter, potted plant, tv, microwave, oven, toaster, sink, refrigerator, clock | Fixed things and appliances stay where they are. |
| **play** | Stays in place | frisbee, snowboard, sports ball, kite, skateboard | In-play things are mid-air or on the ground; they only move with the person while held. |

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

| Object | COCO family | Schemas | Typical size | Closest analogues | Look-alikes |
|---|---|---|---|---|---|
| person | person | — | 1–1.95 m | sheep, teddy bear, cow | — |
| bicycle | vehicle | ride | 1.5–1.9 m | motorcycle, surfboard, skis, horse, elephant | motorcycle |
| car | vehicle | enclose | 3.8–5 m | bus, truck, train, airplane, boat | truck, bus |
| motorcycle | vehicle | ride | 1.8–2.3 m | bicycle, surfboard, horse, skis, elephant | bicycle |
| airplane | vehicle | enclose | 10–70 m | bus, train, truck, car, boat | — |
| bus | vehicle | enclose | 9–13 m | train, airplane, truck, car, boat | car, truck |
| train | vehicle | enclose | 15–30 m | bus, airplane, truck, car, boat | — |
| truck | vehicle | enclose | 5–12 m | bus, train, car, airplane, boat | car, bus |
| boat | vehicle | enclose | 2–15 m | train, bus, truck, airplane, car | — |
| traffic light | outdoor | fixture | 0.6–1.2 m | stop sign, parking meter, toaster, sink, fire hydrant | stop sign, parking meter |
| fire hydrant | outdoor | fixture | 0.6–0.9 m | oven, microwave, traffic light, stop sign, parking meter | — |
| stop sign | outdoor | fixture | 0.6–2.5 m | traffic light, parking meter, potted plant, refrigerator, toaster | traffic light, parking meter |
| parking meter | outdoor | fixture | 1.2–1.6 m | traffic light, stop sign, tv, clock, toaster | stop sign, traffic light |
| bench | outdoor | seat | 1.2–2.5 m | couch, bed, chair, toilet, dining table | chair, couch, toilet, bed, dining table |
| bird | animal | grasp, companion | 0.1–0.8 m | cat, zebra, dog, giraffe, bear | kite, frisbee |
| cat | animal | grasp, companion | 0.3–0.7 m | dog, bird, sheep, cow, zebra | dog, teddy bear, bear |
| dog | animal | companion, grasp | 0.3–1.1 m | cat, bird, horse, sheep, elephant | sheep, cow, cat, teddy bear, bear |
| horse | animal | ride, companion | 1.6–2.5 m | elephant, sheep, cow, dog, motorcycle | cow, zebra |
| sheep | animal | companion | 0.9–1.4 m | cow, elephant, zebra, giraffe, horse | dog, cow |
| cow | animal | companion | 1.8–2.5 m | sheep, giraffe, zebra, elephant, horse | horse, zebra, sheep, dog |
| elephant | animal | ride, companion | 3–6.5 m | horse, giraffe, sheep, cow, zebra | — |
| bear | animal | companion | 1–2.5 m | zebra, giraffe, cow, sheep, bird | dog, cat, teddy bear |
| zebra | animal | companion | 2–2.6 m | bear, giraffe, cow, sheep, bird | horse, cow |
| giraffe | animal | companion | 4–5.5 m | zebra, bear, elephant, cow, sheep | — |
| backpack | accessory | grasp, wear | 0.35–0.6 m | handbag, tie, baseball glove, tennis racket, bottle | handbag, suitcase |
| umbrella | accessory | grasp | 0.6–1.3 m | book, scissors, hair drier, cell phone, laptop | — |
| handbag | accessory | grasp, wear | 0.2–0.5 m | backpack, tie, baseball glove, tennis racket, bottle | backpack, suitcase |
| tie | accessory | grasp, wear | 0.3–0.6 m | backpack, handbag, baseball glove, tennis racket, vase | — |
| suitcase | accessory | grasp | 0.5–0.8 m | teddy bear, book, backpack, umbrella, tennis racket | handbag, backpack |
| frisbee | sports | grasp, play | 0.2–0.3 m | sports ball, remote, scissors, hair drier, toothbrush | kite, bird |
| skis | sports | wear, ride | 1.5–2 m | snowboard, surfboard, bicycle, backpack, skateboard | skateboard, snowboard, surfboard |
| snowboard | sports | wear, ride, play | 1.3–1.7 m | skis, skateboard, surfboard, bicycle, kite | skateboard, surfboard, skis |
| sports ball | sports | grasp, play | 0.06–0.3 m | frisbee, tennis racket, baseball bat, skateboard, donut | apple, orange |
| kite | sports | play | 0.5–2.5 m | snowboard, sports ball, frisbee, skateboard, tennis racket | bird, frisbee |
| baseball bat | sports | grasp | 0.75–0.9 m | tennis racket, teddy bear, remote, keyboard, book | tennis racket |
| baseball glove | sports | grasp, wear | 0.25–0.35 m | backpack, handbag, tie, mouse, remote | — |
| skateboard | sports | grasp, ride, play | 0.7–0.85 m | snowboard, surfboard, frisbee, sports ball, skis | snowboard, surfboard, skis |
| surfboard | sports | ride | 1.7–3 m | bicycle, skis, motorcycle, skateboard, snowboard | skateboard, snowboard, skis |
| tennis racket | sports | grasp | 0.6–0.72 m | baseball bat, bottle, teddy bear, remote, handbag | baseball bat |
| bottle | kitchen | grasp | 0.15–0.4 m | tennis racket, cup, book, backpack, spoon | cup, wine glass, vase, bowl |
| wine glass | kitchen | grasp | 0.13–0.25 m | spoon, fork, bowl, cup, knife | cup, bottle, vase, bowl |
| cup | kitchen | grasp | 0.07–0.16 m | bottle, wine glass, spoon, tennis racket, toothbrush | wine glass, bottle, vase, bowl |
| fork | kitchen | grasp | 0.15–0.22 m | knife, spoon, bowl, wine glass, toothbrush | knife, spoon, toothbrush, scissors |
| knife | kitchen | grasp | 0.15–0.35 m | fork, spoon, bowl, wine glass, scissors | fork, spoon, toothbrush, scissors |
| spoon | kitchen | grasp | 0.12–0.2 m | wine glass, fork, bowl, knife, toothbrush | fork, knife, toothbrush, scissors |
| bowl | kitchen | grasp | 0.1–0.35 m | spoon, fork, wine glass, knife, broccoli | cup, wine glass, bottle, vase |
| banana | food | grasp | 0.12–0.25 m | apple, orange, carrot, donut, sandwich | — |
| apple | food | grasp | 0.06–0.1 m | banana, orange, carrot, broccoli, donut | orange, sports ball |
| sandwich | food | grasp | 0.1–0.25 m | hot dog, carrot, donut, pizza, cake | hot dog, pizza |
| orange | food | grasp | 0.06–0.1 m | apple, banana, carrot, broccoli, sandwich | apple, sports ball |
| broccoli | food | grasp | 0.1–0.25 m | carrot, bowl, apple, banana, sandwich | — |
| carrot | food | grasp | 0.1–0.3 m | broccoli, sandwich, hot dog, banana, apple | — |
| hot dog | food | grasp | 0.12–0.25 m | sandwich, carrot, donut, pizza, cake | sandwich, pizza |
| pizza | food | grasp | 0.2–0.5 m | donut, cake, sandwich, hot dog, banana | donut, cake, sandwich, hot dog |
| donut | food | grasp | 0.07–0.13 m | pizza, sandwich, hot dog, banana, cake | cake, pizza |
| cake | food | grasp | 0.12–0.45 m | pizza, sandwich, hot dog, donut, vase | donut, pizza |
| chair | furniture | seat | 0.75–1.2 m | couch, bed, bench, toilet, surfboard | bench, couch, toilet, bed |
| couch | furniture | seat | 1.4–2.6 m | bed, chair, bench, toilet, dining table | chair, bench, toilet, bed |
| potted plant | furniture | fixture | 0.2–1.8 m | stop sign, toaster, refrigerator, clock, parking meter | — |
| bed | furniture | seat | 1.9–2.3 m | couch, bench, chair, toilet, dining table | chair, bench, couch, toilet |
| dining table | furniture | table | 0.8–2.8 m | bed, couch, bench, toilet, chair | bench |
| toilet | furniture | seat | 0.55–0.85 m | bed, chair, couch, bench, dining table | chair, bench, couch, bed |
| tv | electronic | fixture | 0.5–1.9 m | parking meter, sink, toaster, traffic light, oven | laptop |
| laptop | electronic | grasp | 0.28–0.42 m | keyboard, mouse, book, cell phone, vase | tv |
| mouse | electronic | grasp | 0.08–0.13 m | hair drier, laptop, remote, cell phone, bowl | cell phone, remote |
| remote | electronic | grasp | 0.12–0.25 m | tennis racket, mouse, scissors, hair drier, toothbrush | cell phone, mouse |
| keyboard | electronic | grasp | 0.3–0.5 m | laptop, tennis racket, teddy bear, mouse, vase | — |
| cell phone | electronic | grasp | 0.11–0.18 m | book, mouse, hair drier, laptop, remote | remote, mouse |
| microwave | appliance | fixture | 0.4–0.6 m | oven, toaster, refrigerator, sink, fire hydrant | oven, toaster, refrigerator |
| oven | appliance | fixture | 0.55–0.95 m | microwave, toaster, refrigerator, sink, fire hydrant | microwave, toaster, refrigerator |
| toaster | appliance | fixture | 0.2–0.4 m | oven, microwave, clock, sink, potted plant | microwave, oven, refrigerator |
| sink | appliance | fixture | 0.4–1 m | oven, microwave, toaster, refrigerator, tv | — |
| refrigerator | appliance | fixture | 1.4–2 m | oven, microwave, sink, stop sign, potted plant | microwave, oven, toaster |
| book | indoor | grasp | 0.12–0.35 m | scissors, cell phone, teddy bear, hair drier, toothbrush | — |
| clock | indoor | fixture | 0.15–0.6 m | toaster, parking meter, oven, potted plant, sink | — |
| vase | indoor | grasp | 0.12–0.7 m | keyboard, teddy bear, cake, mouse, scissors | cup, wine glass, bottle, bowl |
| scissors | indoor | grasp | 0.12–0.25 m | book, hair drier, toothbrush, knife, remote | fork, knife, spoon, toothbrush |
| teddy bear | indoor | grasp | 0.15–0.9 m | tennis racket, book, keyboard, vase, scissors | dog, cat, bear |
| hair drier | indoor | grasp | 0.18–0.3 m | mouse, scissors, toothbrush, book, remote | — |
| toothbrush | indoor | grasp | 0.15–0.2 m | spoon, scissors, hair drier, fork, bowl | fork, knife, spoon, scissors |

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
