import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tmovers import ants, configspace, geometry, ppo
from tmovers.colony import (ACTION_DIM, ColonyConfig, ColonyEnv, greedy_policy,
                            paper_policy)


@pytest.fixture(scope="module")
def cspace():
  return configspace.ConfigSpace("large")


@pytest.fixture(scope="module")
def env():
  return ColonyEnv()


@pytest.fixture(scope="module")
def env_all():
  return ColonyEnv(ColonyConfig(controlled="all"))


def test_collision_check():
  spec = geometry.SPECS["large"]
  free = configspace.collides(spec, np.array([
      geometry.initial_pose(spec),  # start: free
      [spec.slit1, 2.0, 0.0],  # inside the slit-1 wall
      [spec.slit1, spec.breadth / 2, np.pi / 2],  # long crossbar across slit
  ]))
  assert free.tolist() == [False, True, True]


def test_distance_to_goal(cspace):
  spec = cspace.spec
  start = float(cspace.distance(jnp.asarray(geometry.initial_pose(spec))))
  # Grid-converged optimum is ~49.2 cm (7.9 d_cor) for the large puzzle.
  assert 47 < start < 52
  goal = jnp.array([spec.slit2 + spec.load_length, spec.breadth / 2, 0.0])
  assert float(cspace.distance(goal)) == 0.0
  # The distance can drop by at most the distance moved (1-Lipschitz).
  q = jnp.asarray(geometry.initial_pose(spec))
  dq = jnp.array([1.0, 0.0, 0.0])
  assert abs(float(cspace.distance(q + dq) - cspace.distance(q))) <= 1.0 + 0.3


def test_optimal_path_follows_paper_states(cspace):
  # Following the distance field downhill visits the paper's optimal
  # solution a -> c -> e -> f -> h (e may be its mirror image e').
  import itertools
  g, D, free = cspace.grid, cspace.dist_np, cspace.free
  nx, ny, nt = g.shape
  i = tuple(int(v) for v in cspace._index(
      jnp.asarray(geometry.initial_pose(cspace.spec))))
  offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
  path = [i]
  while D[i] > 0:
    nbrs = [(min(max(i[0] + a, 0), nx - 1), min(max(i[1] + b, 0), ny - 1),
             (i[2] + c) % nt) for a, b, c in offsets]
    j = min((n for n in nbrs if free[n]), key=lambda n: D[n])
    assert D[j] < D[i]
    i = j
    path.append(i)
  poses = np.array([[g.x0 + g.dx * a, g.y0 + g.dx * b, g.dtheta * c]
                    for a, b, c in path])
  states = np.asarray(configspace.classify(cspace.spec, jnp.asarray(poses)))
  seq = [configspace.STATES[s] for k, s in enumerate(states)
         if k == 0 or s != states[k - 1]]
  assert [s.rstrip("'") for s in seq] == ["a", "c", "e", "f", "h"]


def test_population_update_respects_rates():
  p = ants.AntParams()
  status = jnp.zeros(100_000, jnp.int32)
  near = jnp.zeros_like(status, bool)
  status = ants.population_update(jax.random.PRNGKey(0), status, near, 1.0, p)
  frac = float(jnp.mean(status == ants.INFORMED))
  assert abs(frac - (1 - np.exp(-p.k_on))) < 0.002
  # Nothing attaches next to walls.
  status = ants.population_update(jax.random.PRNGKey(1), status * 0,
                                  jnp.ones_like(near), 1e3, p)
  assert int(jnp.sum(status != ants.EMPTY)) == 0


def test_colony_step_shapes_and_masks(env):
  state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
  assert state.obs.shape == (2, env.n_agents, env.obs_dim)
  assert state.global_state.shape == (2, env.global_dim)
  step = jax.jit(jax.vmap(env.step))
  act = jnp.zeros((2, env.n_agents, ACTION_DIM))
  for _ in range(300):  # 30 s
    state = step(state, act)
  assert bool(jnp.isfinite(state.obs).all())
  status = np.asarray(state.env.status)
  # The policy controls exactly the attached, uninformed ants.
  uninformed = (status == ants.PULLER) | (status == ants.LIFTER)
  np.testing.assert_array_equal(np.asarray(state.mask), uninformed)
  assert uninformed.sum() > 0
  assert float(jnp.abs(state.obs * ~state.mask[..., None]).max()) == 0.0
  # Informed ants follow the paper's rule and always pull.
  informed = status == ants.INFORMED
  assert np.asarray(state.pull)[informed].all()


def test_lifting_ants_do_not_move_the_load(env_all):
  env = env_all
  state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
  step = jax.jit(jax.vmap(env.step))
  lift = jnp.zeros((2, env.n_agents, ACTION_DIM))
  for _ in range(100):
    state = step(state, lift)
  # Only newly attached ants pull, until their next decision (<= 0.1 s).
  moved = jnp.linalg.norm(state.env.data.qpos[:, :2] -
                          env.base.initial_pose[:2], axis=-1)
  assert float(moved.max()) < 0.05


def test_keeping_the_angle(env_all):
  env = env_all
  state = env.reset(jax.random.PRNGKey(0))
  step = jax.jit(env.step)
  n = env.n_agents
  aim = jnp.stack([jnp.ones(n), jnp.ones(n), jnp.full(n, 0.5)], -1)
  for _ in range(50):
    state = step(state, aim)
  keep = aim.at[:, 1].set(0.0).at[:, 2].set(-1.0)
  before = state
  state = step(state, keep)
  # Ants attached before and after keep their angle when not reorienting.
  stayed = np.asarray(before.mask & state.mask)
  np.testing.assert_allclose(np.asarray(state.env.phi)[stayed],
                             0.5 * float(env.phi_max), atol=1e-6)


def test_paper_policy_actions(env):
  state = env.reset(jax.random.PRNGKey(0))
  step = jax.jit(env.step)
  for i in range(200):
    state = step(state, paper_policy(env, state, jax.random.PRNGKey(i)))
  act = np.asarray(paper_policy(env, state, jax.random.PRNGKey(0)))
  assert act.shape == (env.n_agents, ACTION_DIM)
  assert set(np.unique(act[:, :2])) <= {0.0, 1.0}
  assert np.abs(act[:, 2]).max() <= 1.0
  # Lifters never reorient (only pullers re-aim, as in the paper).
  assert not ((act[:, 0] == 0) & (act[:, 1] == 1)).any()


def test_reward_is_progress(env):
  state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
  step = jax.jit(jax.vmap(env.step))
  greedy = jax.jit(jax.vmap(lambda s, k: greedy_policy(env, s, k)))
  total = 0.0
  for i in range(100):
    state = step(state, greedy(state, jax.random.split(jax.random.PRNGKey(i), 2)))
    total = total + state.reward
  # Rewards telescope: total reward = distance reduction in d_cor.
  expected = (env.d_start - state.dist) / env.cspace.d_cor
  np.testing.assert_allclose(total, expected, atol=1e-4)
  assert (np.asarray(state.dist) < env.d_start).all()  # greedy starts well


def test_ppo_update_runs(env):
  cfg = ppo.PPOConfig(num_envs=2, rollout_len=4, minibatches=2, epochs=1,
                      hidden=16)
  init, update = ppo.make_train(env, cfg)
  ts = jax.jit(init)(jax.random.PRNGKey(0))
  ts, metrics = jax.jit(update)(ts, None)
  assert all(np.isfinite(float(v)) for v in metrics.values())
  policy = ppo.make_policy(ts.actor, cfg.hidden)
  action = policy(jax.tree.map(lambda x: x[0], ts.env_state),
                  jax.random.PRNGKey(0))
  assert action.shape == (env.n_agents, ACTION_DIM)
