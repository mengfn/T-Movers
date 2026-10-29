"""Puzzle geometry for the ant "piano-movers" T-load puzzle.

Dimensions follow Dreyer et al., PNAS 2024 (doi:10.1073/pnas.2414274121),
SI Appendix Table S2 (simulated ant arenas) and Fig. S1a. Units are cm.

World frame: x points from the 1st chamber towards the 3rd chamber, the arena
spans y in [0, breadth], and the left boundary's inner face is at x = 0. Slit
walls are centred on x = slit1 and x = slit2, each with a gap of `exit_size`
centred on y = breadth / 2. The 3rd chamber is open on its right side.

Load frame: origin at the centre of mass, stem along +x, long crossbar at the
-x end, short crossbar at the +x end (theta = 0 points the short crossbar
towards the slits, as in the initial configuration).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PuzzleSpec:
  breadth: float
  exit_size: float
  slit1: float
  slit2: float
  wall_thickness: float
  load_length: float  # full extent of the load along its stem
  load_width: float  # thickness of the stem and both crossbars
  long_crossbar: float
  short_crossbar: float
  n_sites: int  # N_max, attachment sites on the load's circumference


# Table S2, "with skirt" columns: loads expanded by a thin rim (finite-size
# correction) that gave quantitative agreement with experiments.
SPECS = {
    "large": PuzzleSpec(
        breadth=19.210, exit_size=3.690, slit1=12.920, slit2=19.160,
        wall_thickness=0.290, load_length=9.6137, load_width=0.6137,
        long_crossbar=4.6137, short_crossbar=2.2370, n_sites=120),
    "small": PuzzleSpec(
        breadth=2.410, exit_size=0.470, slit1=1.615, slit2=2.395,
        wall_thickness=0.028, load_length=1.2137, load_width=0.0887,
        long_crossbar=0.5887, short_crossbar=0.2887, n_sites=15),
}


def load_boxes(spec: PuzzleSpec):
  """Returns (centres, half_sizes) of the load's 3 boxes, relative to its COM."""
  L, w = spec.load_length, spec.load_width
  centres = np.array([[w / 2, 0.0], [L / 2, 0.0], [L - w / 2, 0.0]])
  half = np.array([
      [w / 2, spec.long_crossbar / 2],  # long crossbar
      [L / 2 - w, w / 2],  # stem
      [w / 2, spec.short_crossbar / 2],  # short crossbar
  ])
  areas = 4 * half[:, 0] * half[:, 1]
  com = (centres * areas[:, None]).sum(0) / areas.sum()
  return centres - com, half, com


def load_outline(spec: PuzzleSpec):
  """Counter-clockwise outline of the load polygon, relative to its COM."""
  L, w = spec.load_length, spec.load_width
  a, b, h = spec.long_crossbar / 2, spec.short_crossbar / 2, w / 2
  pts = np.array([
      (0, -a), (w, -a), (w, -h), (L - w, -h), (L - w, -b), (L, -b),
      (L, b), (L - w, b), (L - w, h), (w, h), (w, a), (0, a),
  ])
  _, _, com = load_boxes(spec)
  return pts - com


def attachment_sites(spec: PuzzleSpec):
  """N_max equally spaced sites on the load outline and their outward normals.

  Sites sit at the midpoints of equal arc-length segments, so none lands
  exactly on a corner where the normal would be ambiguous.
  """
  pts = load_outline(spec)
  edges = np.roll(pts, -1, axis=0) - pts
  lengths = np.linalg.norm(edges, axis=1)
  cum = np.concatenate([[0.0], np.cumsum(lengths)])
  s = (np.arange(spec.n_sites) + 0.5) * cum[-1] / spec.n_sites
  edge = np.searchsorted(cum, s, side="right") - 1
  frac = (s - cum[edge]) / lengths[edge]
  pos = pts[edge] + frac[:, None] * edges[edge]
  tangent = edges[edge] / lengths[edge, None]
  normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)  # CCW -> outward
  return pos, normal


def wall_boxes(spec: PuzzleSpec):
  """Returns (centres, half_sizes) of the static wall boxes."""
  B, t = spec.breadth, spec.wall_thickness
  x_end = spec.slit2 + spec.load_length  # 3rd chamber, open on its right
  gap_lo, gap_hi = (B - spec.exit_size) / 2, (B + spec.exit_size) / 2
  boxes = [
      ((x_end - t) / 2, -t / 2, (x_end + t) / 2, t / 2),  # bottom
      ((x_end - t) / 2, B + t / 2, (x_end + t) / 2, t / 2),  # top
      (-t / 2, B / 2, t / 2, B / 2 + t),  # left
  ]
  for x in (spec.slit1, spec.slit2):
    boxes.append((x, gap_lo / 2, t / 2, gap_lo / 2))
    boxes.append((x, (gap_hi + B) / 2, t / 2, (B - gap_hi) / 2))
  boxes = np.array(boxes)
  return boxes[:, :2], boxes[:, 2:]


def initial_pose(spec: PuzzleSpec):
  """Load in state a: centred in the 1st chamber, short crossbar towards +x."""
  return np.array([spec.slit1 / 2, spec.breadth / 2, 0.0])
