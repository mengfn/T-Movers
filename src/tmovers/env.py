"""Batched MJX environment for the ant T-load puzzle.

All methods are pure JAX functions of an `EnvState` for a single puzzle, so
they compose with jax.vmap (many puzzles) and jax.jit / lax.scan (many steps):

  env = TMoversEnv("large")
  state = jax.vmap(env.reset)(jax.random.split(key, 4096))
  state = jax.jit(jax.vmap(env.step))(state)
"""

import dataclasses
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from tmovers import ants, geometry, model


class EnvState(NamedTuple):
  data: mjx.Data
  status: jax.Array  # (N_max,) int32 site status, see ants.EMPTY etc.
  phi: jax.Array  # (N_max,) pull angle relative to the site's outward normal
  key: jax.Array
  time: jax.Array
  solved: jax.Array
  done: jax.Array


class TMoversEnv:

  def __init__(self,
               arena: str = "large",
               dt: float = 0.01,
               params: ants.AntParams = ants.AntParams(),
               max_time: float = 1800.0,
               n_sites: Optional[int] = None):
    # n_sites overrides the arena's N_max (group-size experiments); the load's
    # mass and damping follow it, as in the paper (SI Note 1).
    self.spec = geometry.SPECS[arena]
    if n_sites is not None:
      self.spec = dataclasses.replace(self.spec, n_sites=n_sites)
    self.params = params
    self.dt = dt
    self.max_time = max_time
    self.n_sites = self.spec.n_sites

    self.mj_model = model.build_model(self.spec, dt)
    self.mjx_model = mjx.put_model(self.mj_model)

    # I_zz / m_T about the COM; body_inertia is given in principal axes.
    body = self.mj_model.body("load")
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, body.iquat)
    rot = rot.reshape(3, 3)
    inertia_zz = (rot @ np.diag(body.inertia) @ rot.T)[2, 2]
    self.inertia_ratio = float(inertia_zz / body.mass[0])

    site_pos, site_normal = geometry.attachment_sites(self.spec)
    self.site_pos = jnp.asarray(site_pos)
    self.site_normal = jnp.asarray(site_normal)
    self.outline = jnp.asarray(geometry.load_outline(self.spec))
    wall_c, wall_h = geometry.wall_boxes(self.spec)
    self.wall_c, self.wall_h = jnp.asarray(wall_c), jnp.asarray(wall_h)
    self.initial_pose = jnp.asarray(geometry.initial_pose(self.spec))

  def reset(self, key: jax.Array,
            pose: Optional[jax.Array] = None) -> EnvState:
    """Starts a puzzle with an empty load, by default in the initial state."""
    data = mjx.make_data(self.mjx_model)
    pose = self.initial_pose if pose is None else pose
    data = data.replace(qpos=pose)
    zeros = jnp.zeros(self.n_sites)
    return EnvState(data=data,
                    status=jnp.full(self.n_sites, ants.EMPTY, jnp.int32),
                    phi=zeros,
                    key=key,
                    time=jnp.zeros(()),
                    solved=jnp.zeros((), bool),
                    done=jnp.zeros((), bool))

  def step(self, state: EnvState,
           pull_angles: Optional[jax.Array] = None) -> EnvState:
    """Advances the carriers and the load by one time step.

    Args:
      state: current state.
      pull_angles: optional (N_max,) values in [-1, 1] that replace the model's
        pull directions (as a fraction of phi_max from the site normal) for
        every site that currently holds a puller. Attachment, role switching
        and detachment still follow the paper's rules. With None, the full
        rule-based model of the paper is used.
    """
    p = self.params
    data = state.data
    frame = self.site_frame(data)
    near_wall = self.wall_distance(frame.pos) < p.wall_eps

    n_att = jnp.sum(state.status != ants.EMPTY)
    _, d_lin = ants.damping(n_att, self.n_sites, self.inertia_ratio, p)
    key, sub = jax.random.split(state.key)
    status, phi = ants.update(sub, state.status, state.phi, frame, near_wall,
                              d_lin, self.dt, self.spec, p)
    pulling = ants.is_pulling(status)
    if pull_angles is not None:
      override = jnp.clip(pull_angles, -1, 1) * jnp.deg2rad(p.phi_max_deg)
      phi = jnp.where(pulling, override, phi)
    data = self.physics_step(data, frame, status, pulling, phi)
    return self._advance(state, data, status, phi, key)

  def physics_step(self, data: mjx.Data, frame: ants.SiteFrame, status,
                   pulling, phi) -> mjx.Data:
    """Applies the pullers' forces and the carriers' damping for one step."""
    p = self.params
    n_att = jnp.sum(status != ants.EMPTY)
    dof_damping, _ = ants.damping(n_att, self.n_sites, self.inertia_ratio, p)
    force = ants.generalized_force(pulling, phi, frame, p)
    m = self.mjx_model.replace(dof_damping=dof_damping)
    return mjx.step(m, data.replace(qfrc_applied=force))

  def site_frame(self, data: mjx.Data) -> ants.SiteFrame:
    return ants.SiteFrame(self.site_pos, self.site_normal, data.qpos,
                          data.qvel)

  def _advance(self, state: EnvState, data, status, phi, key) -> EnvState:
    time = state.time + self.dt
    solved = state.solved | self._solved(data.qpos)
    return EnvState(data=data,
                    status=status,
                    phi=phi,
                    key=key,
                    time=time,
                    solved=solved,
                    done=solved | (time >= self.max_time))

  def metrics(self, state: EnvState) -> dict:
    """Per-step quantities for logging and analysis."""
    data = state.data
    frame = ants.SiteFrame(self.site_pos, self.site_normal, data.qpos,
                           data.qvel)
    return {
        "qpos": data.qpos,
        "qvel": data.qvel,
        "n_attached": jnp.sum(state.status != ants.EMPTY),
        "n_informed": jnp.sum(state.status == ants.INFORMED),
        "n_pullers": jnp.sum((state.status == ants.INFORMED) |
                             (state.status == ants.PULLER)),
        "order_parameter": ants.order_parameter(state.status, state.phi,
                                                frame),
        "solved": state.solved,
    }

  def load_corners(self, qpos: jax.Array) -> jax.Array:
    """World coordinates of the load's outline vertices."""
    c, s = jnp.cos(qpos[2]), jnp.sin(qpos[2])
    rot = jnp.array([[c, -s], [s, c]])
    return qpos[:2] + self.outline @ rot.T

  def _solved(self, qpos: jax.Array) -> jax.Array:
    """The load lies entirely inside the 3rd chamber."""
    exit_x = self.spec.slit2 + self.spec.wall_thickness / 2
    return jnp.all(self.load_corners(qpos)[:, 0] > exit_x)

  def wall_distance(self, points: jax.Array) -> jax.Array:
    """Distance from each point to the nearest wall box."""
    q = jnp.abs(points[:, None, :] - self.wall_c[None]) - self.wall_h[None]
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return jnp.min(outside + inside, axis=-1)


def rollout(env: TMoversEnv, keys: jax.Array, n_steps: int,
            record_every: int = 100):
  """Runs len(keys) puzzles in parallel with the paper's ant model.

  Returns the final batched state and metrics recorded every `record_every`
  steps, with leading axes (n_steps // record_every, batch).
  """
  step = jax.vmap(env.step)
  metrics = jax.vmap(env.metrics)

  def chunk(state, _):
    state = jax.lax.fori_loop(0, record_every, lambda _, s: step(s), state)
    return state, metrics(state)

  @jax.jit
  def run(keys):
    state = jax.vmap(env.reset)(keys)
    return jax.lax.scan(chunk, state, None, length=n_steps // record_every)

  return run(keys)
