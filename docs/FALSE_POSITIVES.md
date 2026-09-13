# False-Positive Strategy

## Why this is the most important document in the repository

A false negative means a theft goes unnoticed. The store loses the value of one
item.

A false positive means a staff member is sent to confront a shopper who did
nothing wrong. That is a humiliating experience for a person who came in to buy
shampoo, it exposes the store to a discrimination complaint, and it is how a
loss-prevention system gets switched off and never turned on again.

**These costs are not symmetric, and the system is tuned accordingly.** Every
design decision below trades recall for precision, deliberately.

---

## The core rule

> **No single frame, and no single observation, produces an alert.**

All of the following are normal, are individually harmless, and are each
handled explicitly so that none of them alone can escalate risk:

- picking up merchandise
- browsing merchandise
- putting merchandise back
- holding merchandise
- checking a phone
- reaching into a personal pocket
- adjusting clothing
- carrying a purse, backpack or bag
- putting merchandise into a basket or cart
- being temporarily occluded by another shopper, a display, or their own body

---

## Layer 1 — Positive evidence is individually weak

No positive weight comes close to the alert threshold of 85:

| Evidence | Weight | Alone? |
|---|---|---|
| `SHELF_INTERACTION` | +10 | 12% of threshold |
| `STABLE_HAND_ITEM_ASSOCIATION` | +15 | 18% |
| `ITEM_REMOVED_FROM_SHELF` | +10 | 12% |
| `HAND_MOVED_TO_STORAGE_REGION` | +15 | 18% |
| `CONCEALMENT_MOTION_PROFILE` | +10 | 12% |
| `ITEM_DISAPPEARED_NEAR_STORAGE` | +25 | 29% |
| `ITEM_REMAINS_MISSING` | +20 | 24% |

A test asserts this property directly, so it survives future retuning:

```python
def test_no_single_positive_weight_can_trigger_an_alert():
    for evidence_type, weight in config.risk.weights.items():
        if not evidence_type.is_negative:
            assert weight < config.behavior.alert_threshold
```

Reaching 85 requires **six or seven mutually supporting observations** spread
across seconds of video.

---

## Layer 2 — Negative evidence outweighs positive evidence

| Benign explanation | Weight | Effect |
|---|---|---|
| `ITEM_PLACED_IN_CART` | −70 | wipes out a full sequence |
| `ITEM_PLACED_IN_BASKET` | −70 | wipes out a full sequence |
| `NORMAL_PHONE_INTERACTION` | −70 | wipes out a full sequence |
| `ITEM_RETURNED_TO_SHELF` | −60 | wipes out a full sequence |
| `NO_MERCHANDISE_INTERACTION` | −40 | the pocket-reach case |
| `PERSONAL_EFFECT_INTERACTION` | −40 | it is their own bag |
| `ITEM_VISIBLE_IN_HAND` | −30 | nothing is concealed |
| `UNSTABLE_ITEM_TRACK` | −25 | the tracker, not the shopper |
| `LOW_PERSON_TRACK_CONFIDENCE` | −25 | we are not sure who this is |
| `TEMPORARY_OCCLUSION` | −20 | ordinary occlusion |
| `LOW_POSE_CONFIDENCE` | −20 | we cannot see the hands reliably |
| `CAMERA_OCCLUSION` | −20 | we cannot see the scene |
| `SHORT_ACCIDENTAL_OVERLAP` | −15 | a hand passed in front of a product |

Any one of the top four cancels the entire positive case. The score is clamped
at 0, so it cannot go negative, but the raw total is retained for auditing.

---

## Layer 3 — Episodes, not running totals

Evidence is scoped to an **interaction episode**: one shelf approach through to
its resolution. A benign explanation does not merely subtract points — it
**terminates the episode and clears its positive evidence**
(`EvidenceLedger.clear_positive`).

The negative record is deliberately kept for a moment afterwards, so the score
visibly collapses *and* the explanation says why.

```
SHELF_INTERACTION ──▶ ITEM_ASSOCIATED ─┬─▶ ITEM_RETURNED  ──▶ IDLE
                                       ├─▶ ITEM_TO_BASKET ──▶ IDLE
                                       └─▶ ITEM_TO_CART   ──▶ IDLE
```

Without this, a shopper who picks up three items, returns two and buys one
would accumulate positive evidence across all three interactions.

---

## Layer 4 — Time requirements everywhere

| Requirement | Default | Prevents |
|---|---|---|
| `min_person_track_seconds` | 1.0 s | a detector artifact becoming a suspect |
| `min_shelf_interaction_seconds` | 0.35 s | a wrist clipping a polygon while walking past |
| `min_item_association_seconds` | 0.40 s | a hand passing in front of a shelved product |
| `storage_dwell_seconds` | 0.50 s | a hand *crossing* the body counting as a concealment approach |
| `possibly_occluded_after_seconds` | 0.25 s | a one-frame detector miss |
| `occluded_after_seconds` | 0.60 s | a brief occlusion |
| `item_missing_timeout_seconds` | 1.25 s | declaring an item gone too early |
| `min_age_for_stability_seconds` | 0.30 s | reasoning about a flickering track |

`storage_dwell_seconds` deserves particular mention: it was added after the
simulator showed a hand *sweeping across the body* on its way somewhere else
clipping the torso region and reading as a concealment approach. That single
requirement dropped the `TEMPORARY_OCCLUSION` scenario from 57.5 to 35.0. It is
a good example of the class of bug this design is meant to surface early.

---

## The specific false positives, and how each is handled

### Phone use — the hardest case

A shopper pulling a phone from a pocket, looking at it, and putting it back
reproduces the **entire** concealment motion sequence: hand to waist, object
disappears at the waist, object stays gone. Geometry cannot distinguish them.

**Handling:** object identity. If the tracked object associated with the wrist
is classified `PHONE`:

1. `NORMAL_PHONE_INTERACTION` (−70) is raised;
2. `STABLE_HAND_ITEM_ASSOCIATION` is **never** raised for it;
3. the item is dropped as the episode's active item, so no disappearance
   evidence can be raised about it at all;
4. the item tracker refuses to match a phone detection to a merchandise track,
   so the suppression cannot be bypassed by a mid-track class swap.

This is why `cell phone` is not an optional entry in
`inference.keep_classes`, and why removing it is a behavioral change rather
than an optimization.

**Result: `PHONE_INTERACTION` scores 0.0.**

### Reaching into a pocket with no merchandise

**Handling:** `HAND_MOVED_TO_STORAGE_REGION` requires prior merchandise
context. Without it, the movement raises `NO_MERCHANDISE_INTERACTION` (−40)
instead — recording explicitly *why* nobody is being paged, rather than
leaving an absence of evidence.

Proximity alone is also insufficient: an arm hanging at the side sits inside
the waist region permanently, so the engine requires the wrist to have
**approached** (`storage_approach_travel_ratio`) and then **stayed**
(`storage_dwell_seconds`).

**Result: `POCKET_ADJUSTMENT` scores 0.0.**

### Putting an item in a basket or cart

**Handling:** if an item is (or was last seen) inside a basket/cart zone, or
inside a detected `SHOPPING_CART`/`BASKET` box, it is resolved `IN_BASKET` /
`IN_CART` — a **sticky** status that survives the item subsequently
disappearing from view among other goods.

**Result: `ITEM_TO_BASKET` and `ITEM_TO_CART` score 22.0 and terminate.**

### Picking an item up and putting it back

**Handling:** a return is only recognized after the item was genuinely removed
(`ITEM_REMOVED_FROM_SHELF` must be present) — an item sitting on its shelf
while a hand hovers over it has not been "returned" to anything. The return
then branches the episode to `ITEM_RETURNED` and clears the positive evidence.

**Result: `ITEM_PICKUP_RETURN` peaks at 27.5 and terminates benignly.**

### Temporary occlusion

**Handling:** the disappearance ladder, plus a location test. An item that
vanishes **away from any storage region** raises `TEMPORARY_OCCLUSION` (−20),
not disappearance evidence. Reappearing resets the ladder entirely.

**Result: `TEMPORARY_OCCLUSION` peaks at 35.0, well under threshold.**

### Carrying a bag or backpack

**Handling:** bag and backpack storage regions are emitted **only when a bag is
actually detected** overlapping the person. Inventing a "bag area" for every
shopper would manufacture evidence. Objects classified as personal effects
raise `PERSONAL_EFFECT_INTERACTION` (−40) rather than being treated as
merchandise.

### Poor visibility

**Handling:** low pose confidence (−20) and low track confidence (−25) are
themselves negative evidence. When the system cannot see reliably, it says so
and scores down, rather than guessing.

### An unstable item track

**Handling:** an item must reach `min_hits_for_stability` (4 detections) and
`min_age_for_stability_seconds` (0.3 s) before its disappearance means
anything. Below that, `UNSTABLE_ITEM_TRACK` (−25) is raised — a flickering
detection is a far more likely explanation for a "disappearance" than
concealment is.

### No merchandise detector at all

**Handling:** the most important one. With no custom retail model, the system
has zone geometry and wrist kinematics only. That is real evidence but it is
**not sufficient grounds to ask a human to review a shopper**, so the score is
capped at `zone_only_risk_ceiling` (59) against a threshold of 85. The config
layer refuses to start if the ceiling is ever set at or above the threshold.

**Result: `CONCEALMENT_WITHOUT_ITEM_DETECTOR` peaks at 32.5 and cannot alert.**

---

## Layer 5 — Alert gating beyond the score

`RiskEngine.should_alert()` applies a final check: even at a score ≥ 85, an
alert is refused unless both `STABLE_HAND_ITEM_ASSOCIATION` and
`ITEM_DISAPPEARED_NEAR_STORAGE` are present, and the refusal is logged as a
warning.

A score that reaches the threshold without those is a scoring
misconfiguration, not an incident. This is a backstop against someone
retuning weights and accidentally making zone interaction alone sufficient.

---

## Layer 6 — Cooldown

One sustained sequence would otherwise produce an alert on every analysis
frame. A 30-second per-person cooldown applies, overridden only by a genuine
escalation (+10 risk) or a distinct event type — because suppressing a REVIEW
alert that has since become HIGH RISK would hide exactly the escalation staff
need to see.

Cooldown state is dropped when a track is retired, because track ids are
reused and stale state would suppress a genuine alert for a different shopper.

---

## Current measured behavior

From `python scripts/simulate_behavior.py`:

| Scenario | Expected | Peak risk | Result |
|---|---|---|---|
| `NORMAL_BROWSING` | NO ALERT | 10.0 | ✅ LOW |
| `PHONE_INTERACTION` | NO ALERT | **0.0** | ✅ LOW |
| `POCKET_ADJUSTMENT` | NO ALERT | **0.0** | ✅ LOW |
| `ITEM_TO_BASKET` | NO ALERT | 22.0 | ✅ LOW |
| `ITEM_TO_CART` | NO ALERT | 22.0 | ✅ LOW |
| `ITEM_PICKUP_RETURN` | NO ALERT | 27.5 | ✅ LOW |
| `CONCEALMENT_WITHOUT_ITEM_DETECTOR` | NO ALERT | 32.5 | ✅ ELEVATED |
| `TEMPORARY_OCCLUSION` | NO ALERT | 35.0 | ✅ ELEVATED |
| `POSSIBLE_CONCEALMENT` | ALERT | 100.0 | ✅ HIGH RISK |

The gap between the highest benign case (35.0) and the alert threshold (85) is
the safety margin. **Watch it when retuning.** If a benign scenario climbs into
the 60s, the configuration has drifted even if no test has failed yet.

---

## Tuning guidance

**Before changing any weight:**

```bash
python scripts/simulate_behavior.py --verbose --timeline
python -m pytest tests/test_scenarios.py -v
```

**If you get false positives in production:**

1. Pull the incident JSON and read `evidence_breakdown` — it names exactly
   which observations produced the score.
2. If a benign explanation *was* observed but weighted too lightly, raise that
   negative weight.
3. If the benign explanation was **not observed**, that is a detection or zone
   problem, not a scoring problem. Check the zone polygons first
   (see [ZONES.md](ZONES.md)) — a missing basket zone is the most common cause.
4. Raising `alert_threshold` is the blunt instrument. Prefer fixing the
   specific evidence gap; the threshold change hides the next case too.
5. Add a scenario to `simulation/scenarios.py` reproducing it, and a test. The
   case then stays fixed.

**If you get false negatives:**

Resist the urge to lower the threshold. Check in order:

1. Is a merchandise detector configured? Without one, the ceiling at 59 makes
   alerting impossible **by design** — this is the most common cause by far.
2. Are the shelf zones drawn where merchandise actually is?
3. Is pose reaching the shopper? Check `pose_fps` and
   `pose_relevance_distance_ratio` in `/metrics`.
4. Is the item track stable? `UNSTABLE_ITEM_TRACK` in the evidence means the
   detector is flickering — that is a model problem.

---

## What this system deliberately does not do

- **No facial recognition.** Not implemented, not planned, not possible with
  the models used.
- **No identity recognition.** Track ids are per-camera, per-session, and
  reused.
- **No demographic classification** of race, ethnicity, gender or age. No model
  in the pipeline produces such an output, and no risk weight could consume one.
- **No cross-camera re-identification.** A person leaving one camera's view is
  a new, unrelated track elsewhere.
- **No persistent person profile.** Nothing is retained about an individual
  between episodes, let alone between visits.
- **No autonomous action.** The system creates a record for a human to look at.
  It never determines that a theft occurred.

These are not configuration options. They are absent from the architecture.
