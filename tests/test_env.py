import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco import mjx

from tmovers import ants, geometry
from tmovers.env import TMoversEnv, rollout


@pytest.fixture(scope="module")
def env():
  return TMoversEnv("large")


@pytest.mark.parametrize("arena", ["large", "small"])
def test_site_spacing_matches_ant_size(arena):
  # SI Note 1: adjacent carriers are about 3 mm apart.
  spec = geometry.SPECS[arena]
  pts = geometry.load_outline(spec)
  perimeter = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1).sum()
  assert 0.25 < perimeter / spec.n_sites < 0.3


def test_site_normals_point_outwards():
  spec = geometry.SPECS["large"]
  pos, normal = geometry.attachment_sites(spec)
  centres, half, _ = geometry.load_boxes(spec)

  def inside(p):
    return np.any(np.all(np.abs(p[:, None] - centres[None]) < half[None], -1),
                  -1)

  assert inside(pos - 1e-3 * normal).all()
  assert not inside(pos + 1e-3 * normal).any()


def test_initial_pose_fits_in_first_chamber(env):
  corners = np.asarray(env.load_corners(env.initial_pose))
  spec = env.spec
  assert corners[:, 0].min() > 0
  assert corners[:, 0].max() < spec.slit1 - spec.wall_thickness / 2
  assert 0 < corners[:, 1].min() and corners[:, 1].max() < spec.breadth


def test_overdamped_terminal_velocity(env):
  # With implicit joint damping, a constant generalized force settles at
  # velocity = force / damping, for both translation and rotation.
  pose = jnp.array([env.spec.slit1 / 2, env.spec.breadth / 2, 0.0])
  data = env.reset(jax.random.PRNGKey(0), pose).data
  damp, _ = ants.damping(65, env.n_sites, env.inertia_ratio, env.params)
  m = env.mjx_model.replace(dof_damping=damp)
  force = jnp.array([20.0, -10.0, 5.0])
  data = data.replace(qfrc_applied=force)
  step = jax.jit(mjx.step)
  for _ in range(20):
    data = step(m, data)
  np.testing.assert_allclose(data.qvel, force / damp, rtol=1e-3)


def test_walls_block_the_load(env):
  # Push the load straight into the slit-1 wall, entirely below the gap.
  spec = env.spec
  pose = jnp.array([spec.slit1 / 2, spec.breadth / 4, 0.0])
  assert spec.breadth / 4 + spec.long_crossbar / 2 < (spec.breadth -
                                                       spec.exit_size) / 2
  data = env.reset(jax.random.PRNGKey(0), pose).data
  damp, _ = ants.damping(65, env.n_sites, env.inertia_ratio, env.params)
  m = env.mjx_model.replace(dof_damping=damp)
  data = data.replace(qfrc_applied=jnp.array([300.0, 0.0, 0.0]))
  step = jax.jit(mjx.step)
  for _ in range(3000):
    data = step(m, data)
  corners = np.asarray(env.load_corners(data.qpos))
  wall_face = spec.slit1 - spec.wall_thickness / 2
  assert corners[:, 0].max() < wall_face + 0.02
  assert abs(float(data.qvel[0])) < 1e-3
  assert float(env.wall_distance(jnp.asarray(corners)).min()) > -0.02


def test_solved_in_third_chamber(env):
  spec = env.spec
  pose = jnp.array([spec.slit2 + spec.load_length, spec.breadth / 2, 0.0])
  state = env.step(env.reset(jax.random.PRNGKey(0), pose))
  assert bool(state.solved) and bool(state.done)
  assert not bool(env.step(env.reset(jax.random.PRNGKey(0))).solved)


def test_no_ants_no_motion():
  params = dataclasses.replace(ants.AntParams(), k_on=0.0)
  env = TMoversEnv("large", params=params)
  state = env.reset(jax.random.PRNGKey(0))
  step = jax.jit(env.step)
  for _ in range(50):
    state = step(state)
  assert int(jnp.sum(state.status != ants.EMPTY)) == 0
  np.testing.assert_allclose(state.data.qpos, env.initial_pose, atol=1e-6)


def test_batched_rollout_ants_attach_and_move(env):
  keys = jax.random.split(jax.random.PRNGKey(0), 4)
  state, m = rollout(env, keys, n_steps=3000, record_every=1000)
  n_att = np.asarray(m["n_attached"])
  assert n_att.shape == (3, 4)
  assert (n_att[-1] > 15).all()  # towards <N_att>_eq ~ 65 (SI Eq. 12)
  qpos = np.asarray(state.data.qpos)
  assert np.isfinite(qpos).all()
  corners = jax.vmap(env.load_corners)(m["qpos"].reshape(-1, 3))
  depth = jax.vmap(env.wall_distance)(corners)
  assert float(depth.min()) > -0.02  # no corner ever inside a wall
  moved = np.linalg.norm(qpos[:, :2] - np.asarray(env.initial_pose[:2]), axis=1)
  assert (moved > 0.5).all()
  # Different keys give different trajectories.
  assert np.unique(np.round(qpos[:, 0], 4)).size > 1


def test_pull_angle_override(env):
  # All pullers pulling straight out along their normals.
  state = env.reset(jax.random.PRNGKey(0))
  state = state._replace(status=jnp.full(env.n_sites, ants.PULLER))
  state = env.step(state, pull_angles=jnp.zeros(env.n_sites))
  pulling = (state.status == ants.INFORMED) | (state.status == ants.PULLER)
  assert np.all(np.asarray(state.phi)[np.asarray(pulling)] == 0.0)


def test_html_viewer_embeds_trajectory(env):
  import json
  import re

  from tmovers import html
  from tmovers.render import record

  frames = record(env, jax.random.PRNGKey(0), n_steps=200, every=50)
  page = html.render(env, frames, title="test </script>")
  assert "__TMOVERS_DATA__" not in page
  payload = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                      page, re.S).group(1)
  data = json.loads(payload)
  assert data["title"] == "test </script>"
  assert len(data["qpos"]) == len(data["status"]) == 4
  assert all(len(s) == env.n_sites for s in data["status"])


def test_small_arena_contacts_are_stable():
  # The load's mass scales with N_max (SI Note 1); with area-scaled mass the
  # small load chattered against walls and sank 14% of its width into them.
  env = TMoversEnv("small")
  spec = env.spec
  damp, d_lin = ants.damping(8, env.n_sites, env.inertia_ratio, env.params)
  m = env.mjx_model.replace(dof_damping=damp)
  pose = jnp.array([spec.slit1 / 2, spec.breadth / 4, 0.0])
  force = 2 * 8 * env.params.f0
  data = env.reset(jax.random.PRNGKey(0), pose).data.replace(
      qfrc_applied=jnp.array([force, 0.0, 0.0]))
  step = jax.jit(mjx.step)
  speeds, depth = [], 0.0
  for i in range(3000):
    data = step(m, data)
    if i >= 2000:
      speeds.append(float(data.qvel[0]))
      depth = min(depth, float(env.wall_distance(env.load_corners(data.qpos)).min()))
  assert np.sqrt(np.mean(np.square(speeds))) < 0.01 * force / float(d_lin)
  assert -depth < 0.02 * spec.load_width
