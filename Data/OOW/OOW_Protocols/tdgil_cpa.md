# CPA — Closest Point of Approach
Source: https://tdgil.com/cpa-closest-point-of-approach (fetched and saved verbatim as plain text/markdown)

One important point about radar's value is its ability to tell at a glance how close you are to being on a collision course with an object. Whether another vessel, a buoy, or a rock, radar provides visual evidence of a target's track in relation to you. Some simple plotting provides information about how close you will pass and how long you have before it happens.

## Radar Screen Orientation

**Head Up**: always orients the screen to the direction your vessel is heading. Your direction of travel becomes 000. Benefit: all target echoes are exactly relative to your vessel — if the screen paints a target at 090, look directly off the starboard beam, the target will be there. When plotting own vessel's course you will always plot directly on the 000/180 line.

**North Up**: uses a direction sensor to orient the screen to north. 000 on the screen represents North, and a heading flasher (line) shows your vessel's heading. North Up rotates a relative bearing so it aligns with North. Example: on a course of 045, a relative bearing of 090° will paint on the screen at 135°. An object directly off the starboard beam may therefore appear to be behind you while looking at the screen.

**Course Up**: essentially the same as Head Up until you execute a turn; generally treated the same as Head Up mode while calculating CPA and TCPA.

## Relative Bearings

As it pertains to radar, a relative bearing is always in relationship to your vessel's heading, and progresses **clockwise** from 000° to 360°. Notation: 045R is 45° clockwise of your vessel's heading. 270R is 270° clockwise from the heading. **A radar bearing is never cited as counter-clockwise (left of) your current heading.**

## Relative Plot Labels

- R … Your Ship
- M … Other Ship
- M1 … First Plotted Position of Other Ship
- M2, M3 … Later Plotted Positions of Other Ship
- Mx … Planned Position of Other ship at time of your maneuver to avoid
- (If more than one vessel is being tracked, replace M with another letter such as N, O, P, etc.)
- RML … Relative Motion Line
- DRM … Direction of Relative Motion
- SRM … Speed of Relative Motion
- MRM … Miles of Relative Motion
- NRML … New Relative Motion Line
- CPA … Closest Point of Approach

## Plotting the Target's Location

When a target first appears on the radar, use the VRM and EBL tools to determine range and bearing. Plot the range and bearing of the target, the time, and a target identifier (first location: M1).

Every plotted location has 4 data points: target name, time, range, bearing.

Allow time to elapse (a common interval is 6 minutes, though any convenient length may be used) and capture a new range and bearing. Plot as M2.

### 6 Minute Rule

6 minutes is a standard interval because distance traveled in 6 minutes is 1/10 of the speed. E.g., a vessel traveling 0.7 miles over 6 minutes has a speed of 7 knots.

Draw a straight line beginning at M1 through M2 and continuing past the center of the maneuvering board (the center represents your vessel's location). This is the **relative motion line (RML)**. Its direction in degrees is the **direction of relative motion (DRM)**. Draw an arrowhead at the end to avoid later confusing the DRM with its reciprocal.

Measuring the distance between M1 and M2 gives the **speed of relative motion (SRM)**. With the 6-minute rule, multiply the distance by 10.

**If the RML crosses directly through the center, you are on a collision course.** If it passes above or below the center, it does not, and it's time to calculate how close the vessel will be at its closest point of approach, and when.

### Worked example

- M1 at 0900: range 9.0nm, bearing 274R
- M2 at 0906: 8nm, 276R
- M3 at 0912: 6.9nm, 278R

DRM (of M) = 081R. SRM: measured distance M1→M2 = 1.1nm → 6-minute rule gives SRM 11 knots (cross-checked via nomogram using M1→M3: 2.2nm over 12 minutes → also 11 knots).

## Finding the CPA (Direction and Distance)

Once the RML is plotted, scribe a line perpendicular to the RML from the center to find CPA direction; the distance from the center to the RML gives CPA distance. In the worked example: **CPA ≈ 2nm at 352R (8° to port looking over the bow).**

## Time to CPA (TCPA)

**TCPA is the time remaining until the closest point of approach is reached — not an indication that a collision is imminent or already occurring.** In the worked example, TCPA is 48 minutes from point of first contact: measure the distance from the plotted contact to the CPA point (8.8 miles from M1 to CPA), divide by the relative speed (11 knots): 8.8 / 11 = 0.8 hours = 48 minutes. First contact was 0900, so TCPA corresponds to a closest-approach time of 0948.

## Practice problems (from the source page, answers not reproduced here)

Two additional CPA problems are given on the source page for practice: plot the contacts and determine direction to CPA and distance to CPA, with multiple-choice distance options (8.5 nm / 7.3 nm / 6.2 nm and 7 nm / 6.4 nm / 5.8 nm) provided per problem.
