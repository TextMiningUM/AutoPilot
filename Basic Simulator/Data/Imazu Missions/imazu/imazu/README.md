# Imazu scenario implementation

The Imazu problems are encounter geometries, not a uniquely specified
simulation package. Publications differ in their vessel models, safety domains,
coordinate conventions, and sometimes their reproduced constellation figures.
These files therefore identify the particular reference variant they implement
instead of claiming to be the only canonical implementation.

## Reference convention used here

- Target starting positions and headings come from
  `imazu_constellations.csv`, transcribed from the supplied Appendix D table
  based on Sawada et al. (2021) and Zhai et al. (2022).
- Own ship starts at `(0,-6 NM)`, follows course `000 degrees`, and has its goal
  at `(0,+6 NM)`.
- Nominal vessel speed is `11.7 kn`.
- In overtaking cases 3, 7, 15, 20, and 22, target ship 1 travels at `7.8 kn`.
- Target vessels maintain course and speed through the encounter centre and do
  not run collision avoidance.
- Only the own ship runs the collision-avoidance system under test.

The MOOS simulation uses a uniform spatial scale of `6 NM = 500 m`. Its speed
command `20` represents the nominal `11.7 kn`; the slow target therefore uses
`13.333`, preserving the published `7.8 / 11.7` speed ratio. This compression
changes physical time and distance, but preserves the encounter geometry and
relative speeds.

Case 22 combines an overtaking encounter with two crossing contacts: the own
ship approaches slower target 1 from astern while target 2 crosses diagonally
from starboard and target 3 crosses from starboard. It tests whether an
avoidance system can resolve competing multi-vessel constraints and return to
its original route.

The supplied Appendix D rows for cases 15 and 22 are identical. The repository
preserves that source data rather than silently inventing a distinction; any
claimed difference between them requires an additional authoritative source.
