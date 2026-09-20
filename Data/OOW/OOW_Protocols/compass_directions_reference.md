# Directions, Degrees, and the Compass Rose — A Reference for Autopilot Navigation Logic

This document explains the concept of direction and degrees from first principles, then gives the
full 32-point compass table. It's written to be unambiguous enough to hard-code into an autopilot's
navigation logic, including the specific convention mismatches that most commonly cause silent bugs.

---

## 1. What a "direction" actually is

A direction on a horizontal plane (the sea surface, a chart, a radar screen) is an **angle measured
from a reference direction**. Three things must be fixed before a number like "090°" means anything:

1. **The reference direction** (what is 0°?)
2. **The rotational sense** (does the angle increase clockwise or counter-clockwise?)
3. **The reference frame** (is this angle measured relative to the Earth, or relative to the vessel's own bow?)

Get any one of these three wrong and a perfectly correct-looking number points the wrong way. This is
the root cause of the "heading/axis confusion" class of bug: two systems can both use "0 to 360
degrees" and still disagree, because they picked different answers to (1) and (2).

---

## 2. The compass convention (what marine navigation uses)

Marine and aviation navigation universally uses:

- **Reference direction: True North (0°)**
- **Rotational sense: CLOCKWISE**
- Values run from **000° to 359°** (by convention written as three digits, e.g. "005°" not "5°")
- 090° = East, 180° = South, 270° = West, back to 360°/000° = North

This is **not** the convention used in ordinary mathematics/trigonometry, where:

- **Reference direction: positive x-axis (typically "East" on a standard plot, 0°)**
- **Rotational sense: COUNTER-clockwise**
- 90° = "up" (North on a plot), 180° = "left" (West), 270° = "down" (South)

**These two conventions are genuinely different, not just relabeled.** If your autopilot's internal
math uses standard trigonometric functions (`sin`, `cos`, `atan2`) without explicitly converting, it
is extremely easy to silently mix compass-convention input data with math-convention formulas and get
a heading that's rotated and/or mirrored from what was intended. This is very likely the source of any
lingering "heading/axis confusion" in a system whose CPA/TCPA and bearing math was written using raw
trig functions.

### How to convert between them, exactly

If `compass_heading` is in the marine convention (0°=North, clockwise) and you need to feed it into a
standard math `sin`/`cos` call where 0°=East and angles increase counter-clockwise:

```
math_angle_deg = (90 - compass_heading) % 360
```

And the reverse:

```
compass_heading_deg = (90 - math_angle_deg) % 360
```

**However**, a cleaner and less error-prone approach — and the one already used consistently
throughout this project's own code (`narrate.py`, `colreg_llm_bridge_paused.py`,
`evaluate_run.py`) — is to **never convert to math convention at all**, and instead build your own
compass-native vector functions directly:

```python
def compass_vector(heading_deg, magnitude):
    """Returns (x, y) with x=East-positive, y=North-positive, given a
    compass heading (0=North, clockwise) — no East/North convention
    mismatch possible, because sin gives the East component and cos
    gives the North component directly from a compass angle."""
    h = math.radians(heading_deg)
    x = magnitude * math.sin(h)   # East component
    y = magnitude * math.cos(h)   # North component
    return x, y
```

This works because, for a compass angle measured clockwise from North, the sine of the angle gives the
eastward component and the cosine gives the northward component — directly, with no 90-degree offset
needed. **This is the exact formula already used throughout this project.** If your autopilot's x/y
axes are defined as x=East, y=North, this formula needs no conversion step at all — which is precisely
why it's the safest one to standardize on. If your axes are defined some other way (e.g. x=North,
y=East, or a screen-style y-axis pointing "down"), that's a *fourth* convention choice that must also
be fixed and documented, on top of the three listed in Section 1.

---

## 3. True, Magnetic, Compass, and Relative bearing — four different reference frames

The word "bearing" alone is ambiguous until you say which of these four it is:

| Type | Reference (0°) | Notes |
|---|---|---|
| **True bearing** | True North (geographic North Pole) | What a GPS/chart gives you. Stable, does not depend on location. |
| **Magnetic bearing** | Magnetic North (where a compass needle points) | Differs from True North by an amount called **variation**, which changes by location and slowly drifts year to year. |
| **Compass bearing** | Wherever the ship's physical compass actually points | Differs from Magnetic North by **deviation** (local magnetic interference from the ship itself — engines, electronics, steel hull). Rarely relevant for a software autopilot working from GPS/AIS data, but very relevant for a human reading a physical compass. |
| **Relative bearing** | The vessel's own current heading (bow = 0°) | Changes constantly as the vessel turns, even if the target's true bearing hasn't changed. This is what "the contact is at 045° relative" means. |

**For a software autopilot working from GPS/AIS positions, you almost always want True bearing for
absolute reference and Relative bearing for collision-avoidance geometry** (since COLREG rules like
Rule 13's "abaft the beam" test are defined relative to the vessel's own heading, not to True North).
Magnetic/Compass bearings matter for human-readable output (e.g. VHF radio calls referencing a compass
heading) but should not be used internally for geometry calculations, since variation adds an
unnecessary and location-dependent source of error to a calculation GPS already gives you in True terms.

### Converting between True and Relative bearing

```
relative_bearing = (true_bearing_to_target − own_true_heading + 360) mod 360
true_bearing_to_target = (own_true_heading + relative_bearing) mod 360
```

This is the **R + S = T** rule referenced in the earlier navigation-maths drill set: Relative bearing
+ Ship's heading = True bearing (all in the 0–360° clockwise form). Note that this project's own code
uses a **signed** variant of relative bearing (range −180° to +180°, positive = starboard, negative =
port) rather than the raw 0–360° form — both are mathematically equivalent, but mixing them without
converting is another common source of sign errors; see the earlier `nav_maths_drills.json` set for
extensive drilling on exactly this conversion.

---

## 4. Heading vs. Course vs. Track vs. Bearing — four terms that are NOT interchangeable

An autopilot that treats these as synonyms will make real navigation errors, especially in wind or
current:

- **Heading**: the direction the vessel's **bow is pointing**, right now. What the compass reads.
- **Course** (or Course Over Ground, COG): the direction the vessel is **actually moving** over the
  sea/ground, which can differ from heading due to wind, current, or leeway pushing the vessel
  sideways.
- **Track**: the path already sailed, or the intended path to be sailed (a planned route).
- **Bearing**: the direction *to* some other point or object (a waypoint, a contact, a hazard) —
  never the vessel's own direction of travel.

In calm conditions with no current, heading and course are the same. In any real-world condition with
wind or current, they diverge, and an autopilot steering purely by heading (bow direction) without
correcting for course (actual movement) will drift off its intended track. This project's simulated
scenarios have generally assumed heading = course for simplicity (no wind/current modeled), which is a
reasonable simplification for testing COLREG decision logic in isolation, but it is a simplification —
worth flagging explicitly if this autopilot is ever tested against real-world or wind/current-modeled
conditions, since the heading/course distinction becomes operationally important there.

---

## 5. The compass rose: how the 32 named points are built

The named points of the compass are not arbitrary labels — they're built from a small, regular
pattern, which is worth understanding rather than memorizing as 32 unrelated strings:

1. **4 cardinal points** — North, East, South, West — every 90°.
2. **4 intercardinal (ordinal) points** — Northeast, Southeast, Southwest, Northwest — exactly
   halfway between each pair of cardinals (45°, 135°, 225°, 315°).
3. **8 additional points** splitting each cardinal-to-intercardinal gap in half — named by combining
   the two points they sit between, closer one first: e.g. between North (000°) and Northeast (045°)
   sits North-Northeast (022.5°) — "North" first because it's closer to North than to Northeast.
4. **16 more points** ("by" points) splitting every remaining 22.5° gap in half, named as "[nearest
   main point] by [direction toward the next point]" — e.g. between North (000°) and
   North-Northeast (022.5°) sits North by East (011.25°).

That's 4 + 4 + 8 + 16 = 32 points, each exactly 11.25° apart (360° / 32 = 11.25°).

---

## 6. Full 32-point compass table

| Degrees | Direction Name | Abbreviation |
|---|---|---|
| 000.00° | North | N |
| 011.25° | North by East | N by E |
| 022.50° | North-Northeast | NNE |
| 033.75° | Northeast by North | NE by N |
| 045.00° | Northeast | NE |
| 056.25° | Northeast by East | NE by E |
| 067.50° | East-Northeast | ENE |
| 078.75° | East by North | E by N |
| 090.00° | East | E |
| 101.25° | East by South | E by S |
| 112.50° | East-Southeast | ESE |
| 123.75° | Southeast by East | SE by E |
| 135.00° | Southeast | SE |
| 146.25° | Southeast by South | SE by S |
| 157.50° | South-Southeast | SSE |
| 168.75° | South by East | S by E |
| 180.00° | South | S |
| 191.25° | South by West | S by W |
| 202.50° | South-Southwest | SSW |
| 213.75° | Southwest by South | SW by S |
| 225.00° | Southwest | SW |
| 236.25° | Southwest by West | SW by W |
| 247.50° | West-Southwest | WSW |
| 258.75° | West by South | W by S |
| 270.00° | West | W |
| 281.25° | West by North | W by N |
| 292.50° | West-Northwest | WNW |
| 303.75° | Northwest by West | NW by W |
| 315.00° | Northwest | NW |
| 326.25° | Northwest by North | NW by N |
| 337.50° | North-Northwest | NNW |
| 348.75° | North by West | N by W |
| 360.00° | North (wraps back to 000.00°) | N |

---

## 7. Practical notes for an autopilot implementation

- **Always store and compute headings/bearings as True (0–360°, clockwise from North)** internally.
  Convert to Magnetic only at the point of human-readable output, if needed, and never for internal
  geometry.
- **Never mix the signed (−180 to +180) and unsigned (0 to 360) relative bearing forms** without an
  explicit, named conversion step. Pick one as the canonical internal form (this project uses signed)
  and convert at the boundary whenever external data arrives in the other form.
- **Normalize angle differences correctly.** A naive `heading2 - heading1` breaks at the 000°/360°
  wraparound (e.g. 350° to 010° is a 20° turn, not a 340° turn). Always compute angle differences as:
  ```
  diff = (heading2 - heading1 + 540) % 360 - 180
  ```
  which correctly returns a value in (−180°, +180°] regardless of where the wraparound falls. This
  exact formula is already used throughout this project's own `heading_delta()` function.
- **Round only for display, never for storage or intermediate calculation.** The 32-point compass
  names are a human-readable convenience (useful for VHF radio phrasing, e.g. "the contact bears
  approximately Northeast"), not a precision an autopilot should compute with internally — always keep
  full-precision degrees for any actual steering or collision-avoidance math, and only snap to the
  nearest named point when generating human-facing text.
