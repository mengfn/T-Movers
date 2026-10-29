"""Checks that the colony's action interface is faithful to the paper's model.

Runs the paper's rule-based ants (tmovers.env) and the same rules expressed as
a policy through ColonyEnv (colony.paper_policy) on the same number of
puzzles, and compares carrier counts, puller fractions, load speeds, progress
and solves. The two should agree up to sampling noise.

  uv run python scripts/validate_interface.py --puzzles 32 --minutes 15
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from tmovers import ants
from tmovers.colony import ColonyConfig, ColonyEnv, paper_policy


def stats(n_att, n_pull, speed, dist, solved, d_cor):
  """Inputs (T, B), sampled once per second."""
  t_half = n_att.shape[0] // 2  # skip the initial recruitment phase
  return {
      "carriers": n_att[t_half:].mean(),
      "puller fraction": (n_pull[t_half:] / np.maximum(n_att[t_half:], 1)).mean(),
      "speed [cm/s]": speed[t_half:].mean(),
      "progress [d_cor]": (dist[0] - dist.min(0)).mean() / d_cor,
      "solved": solved[-1].mean(),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--puzzles", type=int, default=16)
  parser.add_argument("--minutes", type=float, default=10.0)
  parser.add_argument("--arena", default="large")
  parser.add_argument("--n-sites", type=int, default=None,
                      help="override the arena's N_max (attachment sites)")
  args = parser.parse_args()

  env = ColonyEnv(ColonyConfig(arena=args.arena, n_sites=args.n_sites))
  base, cs = env.base, env.cspace
  per_second = int(round(1.0 / env.config.decision_dt))
  n_sec = int(args.minutes * 60)
  keys = jax.random.split(jax.random.PRNGKey(0), args.puzzles)

  def record(q, qvel, status, pulling, solved):
    return (jnp.sum(status != ants.EMPTY), jnp.sum(pulling),
            jnp.linalg.norm(qvel[:2]), cs.distance(q), solved)

  @jax.jit
  def run_model(keys):
    steps = int(round(1.0 / base.dt))

    def second(s, _):
      s = jax.lax.fori_loop(0, steps, lambda _, s: base.step(s), s)
      return s, record(s.data.qpos, s.data.qvel, s.status,
                       ants.is_pulling(s.status), s.solved)

    s = jax.vmap(base.reset)(keys)
    return jax.lax.scan(jax.vmap(second, in_axes=(0, None)), s, None,
                        length=n_sec)[1]

  @jax.jit
  def run_policy(keys):

    def decide(carry, _):
      s, key = carry
      key, sub = jax.random.split(key)
      act = jax.vmap(lambda st, k: paper_policy(env, st, k))(
          s, jax.random.split(sub, keys.shape[0]))
      return (jax.vmap(env.step)(s, act), key), None

    def second(carry, _):
      carry, _ = jax.lax.scan(decide, carry, None, length=per_second)
      s = carry[0]
      pulling = (s.env.status != ants.EMPTY) & s.pull
      return carry, jax.vmap(record)(s.env.data.qpos, s.env.data.qvel,
                                     s.env.status, pulling, s.env.solved)

    s = jax.vmap(env.reset)(keys)
    return jax.lax.scan(second, (s, keys[0]), None, length=n_sec)[1]

  results = {}
  for name, run in [("paper model", run_model),
                    ("paper policy", run_policy)]:
    out = [np.asarray(x) for x in run(keys)]
    results[name] = stats(*out, cs.d_cor)
    print(f"{name} done", flush=True)

  print(f"\n{args.puzzles} puzzles x {args.minutes} min")
  names = list(results)
  print(f"{'':>18}" + "".join(f"{n:>15}" for n in names))
  for k in results[names[0]]:
    print(f"{k:>18}" + "".join(f"{results[n][k]:15.3f}" for n in names))


if __name__ == "__main__":
  main()
