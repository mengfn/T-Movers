"""Evaluation on the paper's metrics, for colony policies and for the paper's
own rule-based ant model.

Metrics per puzzle: solved, time to solve [s], path length in configuration
space [d_cor] (the paper's Fig. 2A quantity) and the remaining distance to the
goal [d_cor].
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tmovers.colony import ColonyEnv, ColonyState

Policy = Callable[[ColonyState, jax.Array], jax.Array]


def _summarize(env: ColonyEnv, qpos, solved, dt_record):
  """qpos (T, B, 3) and solved (T, B) recorded every dt_record seconds."""
  cs = env.cspace
  first = jnp.argmax(solved, axis=0)
  solved_any = solved.any(0)
  # Path length up to the moment of solving (or the whole episode).
  steps = jnp.arange(qpos.shape[0])[:, None]
  keep = (steps <= jnp.where(solved_any, first, qpos.shape[0]))[1:]
  d = jnp.diff(qpos, axis=0)
  dtheta = jnp.mod(d[..., 2] + jnp.pi, 2 * jnp.pi) - jnp.pi
  seg = jnp.sqrt(d[..., 0]**2 + d[..., 1]**2 + (cs.r_av * dtheta)**2)
  path = jnp.sum(seg * keep, axis=0) / cs.d_cor
  final = jax.vmap(cs.distance)(qpos[-1]) / cs.d_cor
  out = {
      "solved": solved_any,
      "time_to_solve": jnp.where(solved_any, (first + 1) * dt_record, jnp.nan),
      "path_length": path,
      "final_distance": final,
  }
  return jax.tree.map(np.asarray, out)


def evaluate_policy(env: ColonyEnv, policy: Policy, keys: jax.Array,
                    n_decisions: int):
  """Runs len(keys) puzzles with `policy(state, key) -> (N, 2) actions`."""

  def step(carry, _):
    state, key = carry
    key, sub = jax.random.split(key)
    keys = jax.random.split(sub, state.reward.shape[0])
    actions = jax.vmap(policy)(state, keys)
    state = jax.vmap(env.step)(state, actions)
    return (state, key), (state.env.data.qpos, state.env.solved)

  @jax.jit
  def run(keys):
    state = jax.vmap(env.reset)(keys)
    _, (qpos, solved) = jax.lax.scan(step, (state, keys[0]), None,
                                     length=n_decisions)
    q0 = state.env.data.qpos[None]
    return jnp.concatenate([q0, qpos]), solved

  qpos, solved = run(keys)
  dt = env.config.action_repeat * env.config.dt
  return _summarize(env, qpos, jnp.concatenate([solved[:1] & False, solved]),
                    dt)


def make_evaluator(env: ColonyEnv, policy_fn, keys: jax.Array,
                   n_decisions: int):
  """Jitted `evaluate(params) -> metrics` for `policy_fn(params, state, key)`.

  Unlike evaluate_policy, the parameters are an argument, so evaluating a
  changing policy during training compiles only once.
  """
  dt = env.config.action_repeat * env.config.dt

  @jax.jit
  def run(params):

    def step(carry, _):
      state, key = carry
      key, sub = jax.random.split(key)
      actions = jax.vmap(lambda s, k: policy_fn(params, s, k))(
          state, jax.random.split(sub, keys.shape[0]))
      state = jax.vmap(env.step)(state, actions)
      return (state, key), (state.env.data.qpos, state.env.solved)

    state = jax.vmap(env.reset)(keys)
    _, (qpos, solved) = jax.lax.scan(step, (state, keys[0]), None,
                                     length=n_decisions)
    return jnp.concatenate([state.env.data.qpos[None], qpos]), solved

  def evaluate(params):
    qpos, solved = run(params)
    return _summarize(env, qpos,
                      jnp.concatenate([solved[:1] & False, solved]), dt)

  return evaluate


def evaluate_paper_model(env: ColonyEnv, keys: jax.Array, n_decisions: int):
  """The paper's rule-based ants, recorded at the colony's decision rate."""
  base = env.base
  every = env.config.action_repeat

  def chunk(state, _):
    state = jax.lax.fori_loop(0, every, lambda _, s: base.step(s), state)
    return state, (state.data.qpos, state.solved)

  @jax.jit
  def run(keys):
    state = jax.vmap(base.reset)(keys)
    _, (qpos, solved) = jax.lax.scan(jax.vmap(chunk, in_axes=(0, None)),
                                     state, None, length=n_decisions)
    q0 = state.data.qpos[None]
    return jnp.concatenate([q0, qpos]), solved

  qpos, solved = run(keys)
  dt = every * env.config.dt
  return _summarize(env, qpos, jnp.concatenate([solved[:1] & False, solved]),
                    dt)


def report(name: str, m: dict) -> str:
  solved = m["solved"]
  t = m["time_to_solve"][solved]
  path = m["path_length"][solved]
  return (f"{name:>12}: solved {solved.mean():5.1%} ({solved.sum()}/{solved.size})"
          f" | time {np.median(t) if t.size else np.nan:7.1f} s"
          f" | path {np.median(path) if path.size else np.nan:6.1f} d_cor"
          f" | final dist {np.mean(m['final_distance']):5.2f} d_cor")
