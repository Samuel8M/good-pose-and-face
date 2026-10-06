# World model: physics, memory and dreaming

Good Pose & Face edits photos of the physical world, so it reasons with the physical world. This document covers three parts:

1. Physical laws, stored as structure-mapping descriptions ([`physics.ts`](../src/engine/physics.ts)).
2. How a photo situation is mapped onto those laws to decide what changes ([`memory.ts`](../src/engine/memory.ts), [`sme.ts`](../src/engine/sme.ts)).
3. Memory and "dreaming": how the app learns from choices, modelled on current memory research.

Object knowledge (what each of the 80 detectable things *is*) is described in [OBJECT_ANALOGIES.md](OBJECT_ANALOGIES.md).

## 1. Laws as base domains

In structure-mapping, an analogy carries knowledge from a well-understood **base** to a **target**. Here the base domains are physical laws, and the target is the situation in your photo. Each law is a small relational description with a causal chain. For gravity:

```
(moves P) (holds P X) (graspable X)
(cause (holds P X) (supports P X))
(cause (and (supports P X) (moves P)) (moves-with X P))
```

When you swap a person who is holding a cup, the situation (*P moves, P holds the cup, a cup is graspable*) maps onto this law. SME then produces the law's unmatched consequence as a **candidate inference**: *(moves-with cup P)*. So the cup comes along, and the reason shown is the law's.

| Law | Physics | What it decides |
|---|---|---|
| Gravity and support | W = m·g, g ≈ 9.81 m/s². At rest ΣF = 0, so the hand supplies N = W; without it the object accelerates down at g. | Held things come along; left behind they would float. |
| Rigid attachment | Bodies fastened together share one velocity: v_X = v_P. | Worn things (tie, bag, glove) come along. |
| Rider and ride are one system | One system with momentum p = (m_P + m_X)·v. | Bikes, boards, skis and horses come along with their rider. |
| Seats rest on the ground | The floor supplies N = (m_seat + m_person)·g; static friction (≤ μN) keeps the seat in place when the person gets up. | Chairs, benches, couches stay. |
| Nearer things hide farther things | Light travels in straight lines; along each camera ray only the nearest surface (smallest depth Z) is seen. | People and things in front stay on top. |
| Things behind keep their place | Newton's first law: a body at rest stays at rest unless a force acts. | People and things behind stay, giving way only where the new pose covers them. |
| People carry their own weight | Each person is held up by the ground through their own feet (N = m·g). | Touching neighbours stay. |
| Contact that persisted moved together | Things that stayed in contact between shots had the same displacement Δx. | A small thing that stayed with them in the other shot comes along. |
| Big things carry their own weight | A large object is supported by the floor, not by whoever leans on it. | Leaned-on furniture stays. |
| Object permanence | Things keep existing and keep their place unless something moves them. | Anything the new pose merely overlaps stays. |
| The holder is not the held | Swapping what is held changes only the held thing. | The person holding a swapped object stays. |

Two more laws are used numerically rather than symbolically:

- **Perspective (pinhole camera).** Image height h = f·H / Z: things at the same depth Z share one scale. People standing on the same ground line as an object give metres-per-pixel for checking its size ([`knowledge.ts`](../src/engine/knowledge.ts)). Things lower in the frame are nearer, which orders occlusion.
- **Light (Lambert's cosine law).** A matte surface's brightness is I = ρ·E·max(0, n·l), where ρ is albedo, E is the light's irradiance (falling off as E ∝ 1/r²), n is the surface normal, and l is the light direction. Ramamoorthi & Hanrahan (2001) and Basri & Jacobs (2003) showed that diffuse shading under any distant lighting is captured almost entirely by low-order spherical harmonics. The app fits the first-order form I(n) ≈ c₀ + c·n on each face's skin, using normals computed from the face mesh's 3D landmarks. It then multiplies the incoming face by *target shading ÷ donor shading*, so its highlights and shadows fall where the target photo's light puts them. In a test with a physically side-lit donor face, this raised the match to the true face from 17.7 dB to 20.0 dB compared with brightness correction alone. A simple left-to-right brightness ramp is better handled by the existing brightness-plane fit, and both run.

## 2. Deciding what changes

For each thing related to the swapped person, the app builds a structured description of the situation. Here the cup *is held by P*, *P moves*, the cup *is graspable* and *small*, and the same holding relation *persisted* in the other shot. Retrieval then works in two stages, following MAC/FAC (Forbus, Gentner & Law, 1995):

1. **MAC:** a cheap filter keeps only memories sharing relational vocabulary with the situation.
2. **FAC:** SME scores the survivors: physical laws, your generalizations and your episodes. The closest five vote, weighted by similarity squared times trust. Your own choices count three times a law. Remembered choices only vote when they are structurally close (similarity ≥ 0.6), so a lesson about seats never decides what happens to a held cup.

## 3. Memory and dreaming

The design follows **Complementary Learning Systems** theory (McClelland, McNaughton & O'Reilly, 1995; updated by Kumaran, Hassabis & McClelland, 2016). The brain learns with two systems: a fast one (hippocampus) that stores individual episodes at once, and a slow one (neocortex) that gradually extracts structure. Replay of episodes, especially during sleep, transfers knowledge from the first to the second.

- **Episodic memory.** Every time you override a decision, that situation and your choice are stored immediately as a structured description, in `localStorage` under `good-pose-and-face:memory:v2`.
- **Dreaming** runs while the app is idle, in two phases inspired by Lewis, Knoblich & Poe (2018). Their review argues that non-REM replay abstracts rules and REM replay forms novel associations.
  - *Phase 1 (abstraction, like non-REM).* New episodes are replayed in random order. Each is merged into the most similar existing generalization if SME similarity is at least 0.7; otherwise it starts a new generalization with another similar episode. This is the assimilation scheme of **SAGE**, the Sequential Analogical Generalization Engine (Kuehne, Forbus, Gentner & Quinn, 2000; McLure, Friedman & Forbus, 2010). A generalization keeps a count for each statement, giving its probability; statements in at least half the members form its description.
  - *Phase 2 (association, like REM).* Each generalization is mapped onto every physical law. If your choices agree with the best-matching law, the dream records the explanation. If they disagree, it records that your preference wins in similar photos. Each generalization is also extended to analogous objects from [OBJECT_ANALOGIES.md](OBJECT_ANALOGIES.md), such as a chair lesson extending to couches and benches. These findings appear as **insights** under *How does it know what to change?*
- **Forgetting.** *Forget what it learned* clears episodes, generalizations and insights. The physical laws always remain.

## Sources

- Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy. *Cognitive Science*, 7.
- Falkenhainer, B., Forbus, K. & Gentner, D. (1989). The Structure-Mapping Engine. *Artificial Intelligence*, 41.
- Forbus, K., Gentner, D. & Law, K. (1995). MAC/FAC: A model of similarity-based retrieval. *Cognitive Science*, 19.
- Kuehne, S., Forbus, K., Gentner, D. & Quinn, B. (2000). SEQL: Category learning as progressive abstraction using structure mapping. *Proc. Cognitive Science Society*.
- McLure, M., Friedman, S. & Forbus, K. (2010). Learning concepts from sketches via analogical generalization and near-misses. *Proc. Cognitive Science Society*.
- McClelland, J., McNaughton, B. & O'Reilly, R. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102.
- Kumaran, D., Hassabis, D. & McClelland, J. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences*, 20.
- Lewis, P., Knoblich, G. & Poe, G. (2018). How memory replay in sleep boosts creative problem-solving. *Trends in Cognitive Sciences*, 22(6), 491–503.
- Ramamoorthi, R. & Hanrahan, P. (2001). An efficient representation for irradiance environment maps. *SIGGRAPH*.
- Basri, R. & Jacobs, D. (2003). Lambertian reflectance and linear subspaces. *IEEE PAMI*, 25(2).
