"""The load's configuration space (x, y, theta): collisions, distance to goal,
and the puzzle states of SI Appendix Table S3.

Lengths in configuration space follow the paper: a rotation d_theta counts as
r_av * d_theta, where r_av is the average distance from the load's periphery to
its centre of mass (Table S1/S2), so ds^2 = dx^2 + dy^2 + (r_av d_theta)^2.
"""

import hashlib
import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

from tmovers import geometry

# r_av from Table S2 [cm]; it is not affected by the skirt.
R_AV = {"large": 7.30, "small": 0.92}

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tmovers")


def collides(spec: geometry.PuzzleSpec, poses: np.ndarray,
             chunk: int = 100_000) -> np.ndarray:
  """Whether the load at each pose (..., 3) overlaps a wall.

  Separating-axis test between each of the load's 3 oriented boxes and each
  axis-aligned wall box.
  """
  load_c, load_h, _ = geometry.load_boxes(spec)
  wall_c, wall_h = geometry.wall_boxes(spec)
  poses = np.asarray(poses, np.float64)
  flat = poses.reshape(-1, 3)
  out = np.empty(len(flat), bool)
  for start in range(0, len(flat), chunk):
    p = flat[start:start + chunk]
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    u = np.stack([c, s], -1)[:, None, None, :]  # load box axes, (P, 1, 1, 2)
    v = np.stack([-s, c], -1)[:, None, None, :]
    centres = (p[:, None, None, :2] + load_c[None, :, None, 0:1] * u +
               load_c[None, :, None, 1:2] * v)  # (P, 3, 1, 2)
    d = centres - wall_c[None, None]  # (P, 3, W, 2)
    hu = load_h[None, :, None, 0]
    hv = load_h[None, :, None, 1]
    separated = np.zeros(d.shape[:-1], bool)
    for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), u, v):
      r_load = hu * np.abs(np.sum(u * axis, -1)) + hv * np.abs(
          np.sum(v * axis, -1))
      r_wall = (wall_h[:, 0] * np.abs(axis[..., 0]) +
                wall_h[:, 1] * np.abs(axis[..., 1]))
      separated |= np.abs(np.sum(d * axis, -1)) > r_load + r_wall
    out[start:start + chunk] = ~separated.all(axis=(1, 2))
  return out.reshape(poses.shape[:-1])


def is_solved(spec: geometry.PuzzleSpec, poses: np.ndarray) -> np.ndarray:
  """The whole load lies in the 3rd chamber."""
  outline = geometry.load_outline(spec)
  poses = np.asarray(poses)
  c, s = np.cos(poses[..., 2:3]), np.sin(poses[..., 2:3])
  x = poses[..., 0:1] + c * outline[:, 0] - s * outline[:, 1]
  return np.all(x > spec.slit2 + spec.wall_thickness / 2, axis=-1)


@dataclass(frozen=True)
class Grid:
  """Regular grid over (x, y, theta); theta is periodic over [0, 2 pi)."""
  x0: float
  y0: float
  dx: float
  dtheta: float
  shape: tuple

  def poses(self) -> np.ndarray:
    nx, ny, nt = self.shape
    x = self.x0 + self.dx * np.arange(nx)
    y = self.y0 + self.dx * np.arange(ny)
    t = self.dtheta * np.arange(nt)
    return np.stack(np.meshgrid(x, y, t, indexing="ij"), -1)


def make_grid(spec: geometry.PuzzleSpec, dx: float, dtheta_deg: float) -> Grid:
  x_end = spec.slit2 + spec.load_length
  nx = int(np.ceil(x_end / dx)) + 1
  ny = int(np.ceil(spec.breadth / dx)) + 1
  nt = int(round(360 / dtheta_deg))
  return Grid(0.0, 0.0, dx, 2 * np.pi / nt, (nx, ny, nt))


def distance_to_goal(spec: geometry.PuzzleSpec, grid: Grid, r_av: float):
  """Shortest collision-free path length from every grid pose to the goal.

  Dijkstra over the 26-connected grid, from all solved poses at once. Poses in
  collision get the distance of their nearest free cell, so that slightly
  penetrating physics states still read sensibly. Free cells that are cut off
  from the goal (artefacts of discretization) get the largest finite value.
  """
  poses = grid.poses()
  free = ~collides(spec, poses)
  goal = free & is_solved(spec, poses)

  nx, ny, nt = grid.shape
  index = np.arange(free.size).reshape(free.shape)
  rows, cols, weights = [], [], []
  for ox in (-1, 0, 1):
    for oy in (-1, 0, 1):
      for ot in (-1, 0, 1):
        if (ox, oy, ot) <= (0, 0, 0):  # each undirected edge once
          continue
        xs = slice(max(0, -ox), nx - max(0, ox))
        ys = slice(max(0, -oy), ny - max(0, oy))
        a = index[xs, ys, :]
        b = np.roll(index, -ot, axis=2)[
            slice(max(0, ox), nx + min(0, ox)),
            slice(max(0, oy), ny + min(0, oy)), :]
        ok = free.ravel()[a] & free.ravel()[b]
        length = np.sqrt((ox * grid.dx)**2 + (oy * grid.dx)**2 +
                         (ot * grid.dtheta * r_av)**2)
        rows.append(a[ok])
        cols.append(b[ok])
        weights.append(np.full(ok.sum(), length, np.float32))
  graph = sparse.coo_matrix(
      (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
      shape=(free.size, free.size)).tocsr()
  dist = csgraph.dijkstra(graph, directed=False, indices=np.flatnonzero(goal),
                          min_only=True).reshape(free.shape)

  finite = np.isfinite(dist)
  dist[free & ~finite] = dist[finite].max()
  # Fill blocked cells from their nearest free neighbour.
  _, nearest = ndimage.distance_transform_edt(~free, return_indices=True)
  dist = dist[tuple(nearest)]
  return dist.astype(np.float32), free


class ConfigSpace:
  """Distance-to-goal lookups for a puzzle, cached on disk."""

  def __init__(self, arena: str = "large", dx: float | None = None,
               dtheta_deg: float = 2.5):
    self.spec = geometry.SPECS[arena]
    self.r_av = R_AV[arena]
    dx = dx if dx is not None else self.spec.breadth / 96
    self.grid = make_grid(self.spec, dx, dtheta_deg)
    key = hashlib.sha1(repr((self.spec, self.r_av, self.grid)).encode())
    path = os.path.join(_CACHE_DIR, f"dist_{arena}_{key.hexdigest()[:12]}.npz")
    if os.path.exists(path):
      cached = np.load(path)
      dist, free = cached["dist"], cached["free"]
    else:
      dist, free = distance_to_goal(self.spec, self.grid, self.r_av)
      os.makedirs(_CACHE_DIR, exist_ok=True)
      np.savez_compressed(path, dist=dist, free=free)
    self.dist_np, self.free = dist, free
    self.dist = jnp.asarray(dist)
    # Gradient of the distance field [cm/cm, cm/cm, cm/rad]; theta is periodic.
    g = self.grid
    d_x, d_y = np.gradient(dist, g.dx, axis=(0, 1))
    d_t = (np.roll(dist, -1, axis=2) - np.roll(dist, 1, axis=2)) / (2 * g.dtheta)
    self.grad = jnp.asarray(np.stack([d_x, d_y, d_t], -1), jnp.float32)
    # Path lengths are reported in units of the distance between the slits.
    self.d_cor = self.spec.slit2 - self.spec.slit1

  def _index(self, qpos):
    g = self.grid
    ix = jnp.clip(jnp.round((qpos[..., 0] - g.x0) / g.dx), 0, g.shape[0] - 1)
    iy = jnp.clip(jnp.round((qpos[..., 1] - g.y0) / g.dx), 0, g.shape[1] - 1)
    it = jnp.round(jnp.mod(qpos[..., 2], 2 * jnp.pi) / g.dtheta) % g.shape[2]
    return ix.astype(int), iy.astype(int), it.astype(int)

  def _interp(self, table, qpos):
    """Trilinear interpolation of a grid table at qpos (theta periodic)."""
    g = self.grid
    fx = jnp.clip((qpos[..., 0] - g.x0) / g.dx, 0, g.shape[0] - 1.001)
    fy = jnp.clip((qpos[..., 1] - g.y0) / g.dx, 0, g.shape[1] - 1.001)
    ft = jnp.mod(qpos[..., 2], 2 * jnp.pi) / g.dtheta
    ix, iy, it = (jnp.floor(v).astype(int) for v in (fx, fy, ft))
    wx, wy, wt = fx - ix, fy - iy, ft - it
    out = 0.0
    for cx in (0, 1):
      for cy in (0, 1):
        for ct in (0, 1):
          w = ((wx if cx else 1 - wx) * (wy if cy else 1 - wy) *
               (wt if ct else 1 - wt))
          value = table[ix + cx, iy + cy, (it + ct) % g.shape[2]]
          out = out + (w[..., None] if value.ndim > w.ndim else w) * value
    return out

  def distance(self, qpos):
    """Shortest remaining path length to the goal [cm], jit/vmap-friendly."""
    return self._interp(self.dist, qpos)

  def gradient(self, qpos):
    """Gradient of the distance to the goal w.r.t. (x, y, theta)."""
    return self._interp(self.grad, qpos)

  def path_length(self, qpos_traj):
    """Cumulative configuration-space distance along a trajectory (T, 3)."""
    d = jnp.diff(qpos_traj, axis=0)
    dtheta = jnp.mod(d[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
    return jnp.sum(jnp.sqrt(d[:, 0]**2 + d[:, 1]**2 + (self.r_av * dtheta)**2))


# Puzzle states (Table S3) ---------------------------------------------------

STATES = ("a", "b", "c", "e", "e'", "f", "g", "h")


def load_points(spec: geometry.PuzzleSpec, qpos):
  """Points L1..L6 of Fig. S1a in world coordinates, shape (..., 6, 2).

  L1/L3: outer corners of the short crossbar, L2: tip of the stem's short end,
  L4/L6: outer corners of the long crossbar, L5: middle of its outer edge.
  """
  _, _, com = geometry.load_boxes(spec)
  L = spec.load_length
  a, b = spec.long_crossbar / 2, spec.short_crossbar / 2
  local = jnp.array([[L, b], [L, 0.0], [L, -b], [0.0, -a], [0.0, 0.0],
                     [0.0, a]]) - com
  c, s = jnp.cos(qpos[..., 2]), jnp.sin(qpos[..., 2])
  rot = jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2)
  return qpos[..., None, :2] + jnp.einsum("...ij,kj->...ki", rot, local)


def classify(spec: geometry.PuzzleSpec, qpos):
  """Index into STATES for each pose, following Table S3.

  Best-effort reconstruction: Table S3 defines every state except g, so
  configurations matching none of the conditions are labelled g. M3 and M2
  are taken at the centre lines of slit 1 and slit 2.
  """
  p = load_points(spec, qpos)
  x = p[..., 0]
  y = p[..., 1]
  xm3, xm2 = spec.slit1, spec.slit2
  l1, l2, l3, l4, l5, l6 = (x[..., i] for i in range(6))
  del l1, l3
  h = (xm2 < l2) & (xm2 < l5)
  f = (xm2 < l2) & (xm3 < l5) & (l5 < xm2)
  a = (l2 < xm3) & ((l6 < xm3) | (l4 < xm3))
  c = (l2 < xm3) & (xm3 < l4) & (xm3 < l6)
  b = (xm3 < l2) & (l6 < xm3) & (l4 < xm3)
  e_any = (xm3 < l2) & (l5 < xm2)
  e = e_any & (y[..., 1] > y[..., 4])
  e_prime = e_any & ~e
  # Priority order resolves overlaps between the conditions.
  out = jnp.full(x.shape[:-1], STATES.index("g"))
  for name, cond in reversed([("h", h), ("f", f), ("a", a), ("c", c),
                              ("b", b), ("e", e), ("e'", e_prime)]):
    out = jnp.where(cond, STATES.index(name), out)
  return out
