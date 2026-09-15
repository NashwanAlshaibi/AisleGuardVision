# Zone Configuration

## Why zones exist

A COCO-trained YOLO model has **no generic retail-product class**. It cannot
recognize a wig, a hair bundle, a jar of gel, a cosmetics package or most other
store inventory.

So the MVP grounds merchandise interaction in **configured geometry** instead:
you mark where the merchandise is, and the behavior engine reasons about wrists
entering and leaving those polygons. A future custom merchandise model adds
item-level evidence on top without changing any of this.

**Zones are per camera and are defined in that camera's pixel coordinates.**
Re-aiming a camera, changing its resolution, or changing its stream profile
invalidates its zones. Treat zone calibration as part of camera installation.

---

## Zone kinds

| Kind | Purpose | Effect on risk |
|---|---|---|
| `shelf` | A shelf face holding merchandise | Enables `SHELF_INTERACTION` (+10) |
| `high_value` | A section warranting more attention | As shelf, plus `HIGH_VALUE_ZONE_INTERACTION` (+5) and a `risk_multiplier` |
| `basket` | Where shoppers stage baskets | Placement here → `ITEM_PLACED_IN_BASKET` (−70) |
| `cart` | Where carts sit | Placement here → `ITEM_PLACED_IN_CART` (−70) |
| `checkout` | Register area | Reserved for checkout-aware logic |
| `entrance` | Door area | Reserved for exit-without-checkout logic |
| `exclusion` | Mask out a region entirely | Wrists inside are ignored |

`risk_multiplier` scales **positive** contributions only. Standing in a
high-value aisle must never make the evidence in your favour count for less,
and a test asserts that.

---

## Calibrating a camera

### 1. See what the camera sees

```bash
python scripts/test_stream.py --source "$CAM_001_RTSP" --camera-id cam_001
```

This reports the resolution, the real frame rate (not the one the camera
claims), and the backend. Note the resolution — your polygon coordinates must
be in that space.

### 2. Click out a polygon

```bash
python scripts/test_stream.py --source "$CAM_001_RTSP" \
    --camera-id cam_001 --pick-zone --zone-id shelf_001 --zone-kind shelf
```

Left-click to add a vertex, right-click to undo, `w` to print YAML, `q` to
quit. It emits a block you can paste directly:

```yaml
      - id: shelf_001
        kind: shelf
        name: shelf_001
        polygon:
          - [112, 148]
          - [604, 151]
          - [598, 702]
          - [107, 698]
```

### 3. Verify

```bash
python scripts/test_stream.py --source "$CAM_001_RTSP" --camera-id cam_001 --zones
```

The polygons are drawn over live video. Check them with a person standing in
the aisle, not on an empty store.

---

## What makes a good zone

**Cover the merchandise, not the aisle.** The polygon should hug the shelf
face. A zone that extends into the walkway will trigger on everyone who walks
past, and since `SHELF_INTERACTION` is a prerequisite for the whole sequence,
an over-large zone raises the baseline for every shopper.

**Leave the shopper outside it.** In a 2D camera view, a person standing at a
shelf has their own body projected onto the shelf polygon. If the zone extends
to where people stand, their wrist is "in the shelf zone" permanently. Draw the
polygon at the shelf face and let `shelf_approach_distance_ratio` (default 0.12
body heights) handle the reach.

**One zone per shelf face, not one per store.** Separate zones let the engine
detect an item leaving the zone it came from, which is what
`ITEM_REMOVED_FROM_SHELF` depends on. One giant zone means an item never
"leaves" anything.

**Basket and cart zones are worth the effort.** They are the strongest benign
signal available (−70). A store with shelf zones but no basket zone will have a
materially higher false-positive rate, because ordinary shopping looks like a
sequence that never resolves.

**Use exclusion zones for known noise.** A staff doorway, a mirror, a display
screen playing video, or a neighbouring aisle visible at the frame edge.

---

## Complete example

```yaml
cameras:
  - id: cam_001
    name: Beauty Aisle 3
    source: ${CAM_001_RTSP}
    enabled: true

    zones:
      # The shelf face. Hugs the merchandise, stops short of the walkway.
      - id: shelf_001
        kind: shelf
        name: Hair Care Left
        polygon:
          - [112, 148]
          - [604, 151]
          - [598, 702]
          - [107, 698]

      # High-value section: same geometry rules, more weight.
      - id: shelf_002
        kind: high_value
        name: Wig Wall
        risk_multiplier: 1.25
        polygon:
          - [620, 150]
          - [900, 150]
          - [900, 700]
          - [620, 700]

      # Where shoppers put their baskets down. Strongly reduces risk.
      - id: basket_001
        kind: basket
        name: Basket Staging
        polygon:
          - [950, 500]
          - [1250, 500]
          - [1250, 700]
          - [950, 700]

      # Staff doorway: masked out entirely.
      - id: exclusion_001
        kind: exclusion
        name: Staff Doorway
        polygon:
          - [0, 0]
          - [200, 0]
          - [200, 140]
          - [0, 140]
```

Zones may also be declared globally, each naming its `camera_id` — useful when
calibration is generated by tooling and kept in its own file:

```yaml
zones:
  - id: shelf_010
    camera_id: cam_004
    kind: shelf
    polygon: [[100, 100], [500, 100], [500, 600], [100, 600]]
```

---

## Geometry notes

- Polygons need **at least 3 vertices**; a degenerate (zero-area) polygon is
  rejected with an error and that zone is skipped — the camera keeps running.
- **Concave polygons are fully supported.** An L-shaped shelf face works
  correctly; the point-in-polygon test is a crossing-number algorithm, not a
  convex-hull approximation.
- Vertex order does not matter (clockwise and counter-clockwise both work).
- Coordinates are floats but pixel-scaled; sub-pixel precision is irrelevant.
- Point-in-polygon runs vectorized over all query points per zone, so many
  zones cost far less than the loop structure suggests.

---

## Running without zones

The system runs with no zones configured. It will:

- track people and items normally;
- never raise `SHELF_INTERACTION`, so the concealment sequence can never start;
- give every person baseline pose coverage (with nothing to be near, everybody
  gets attention rather than nobody).

`ZoneRegistry.describe()` says so explicitly:

```
camera cam_001: no zones configured (zone-driven evidence disabled)
```

This is a valid configuration for evaluating detection and tracking quality on
new footage before doing calibration work. It is not a useful production
configuration.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Every shopper reaches `SHELF_INTERACTION` | Zone extends into the walkway |
| Nobody ever reaches it | Zone is in the wrong coordinate space — check the resolution matches what `test_stream.py` reported |
| `ITEM_REMOVED_FROM_SHELF` never fires | One oversized zone; split it per shelf face |
| Basket placements are missed | No basket zone configured, or it does not cover where baskets actually sit |
| Risk spikes near a doorway or screen | Add an `exclusion` zone |
| Zones stopped matching after maintenance | The camera was re-aimed or its profile changed; recalibrate |
