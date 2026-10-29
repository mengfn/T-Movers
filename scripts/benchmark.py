"""Measures colony throughput (ant decisions per second), e.g. on a GPU:

  uv run python scripts/benchmark.py --batch 256 1024 4096

One decision is 0.1 s of puzzle time (10 physics steps) for every ant on the
load. Also reports how long one PPO update would take at each batch size.
"""

import argparse
import time

import jax

from tmovers import ppo
from tmovers.colony import ColonyEnv, random_policy


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, nargs="+", default=[64, 256])
  parser.add_argument("--decisions", type=int, default=200)
  parser.add_argument("--rollout", type=int, default=64)
  args = parser.parse_args()

  env = ColonyEnv()
  print(f"devices: {jax.devices()}", flush=True)
  for batch in args.batch:

    @jax.jit
    def run(keys):
      state = jax.vmap(env.reset)(keys)

      def step(s, k):
        act = jax.vmap(lambda st, kk: random_policy(env, st, kk))(
            s, jax.random.split(k, batch))
        return jax.vmap(env.step)(s, act), None

      return jax.lax.scan(step, state, jax.random.split(keys[0],
                                                        args.decisions))[0]

    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    jax.block_until_ready(run(keys))
    start = time.perf_counter()
    jax.block_until_ready(run(jax.random.split(jax.random.PRNGKey(1), batch)))
    rate = batch * args.decisions / (time.perf_counter() - start)
    sim = rate * env.config.decision_dt
    print(f"batch {batch:6d}: {rate:10,.0f} decisions/s "
          f"({sim:9,.0f} simulated s/s) | rollout of {args.rollout} per "
          f"update ~ {batch * args.rollout / rate:6.2f} s", flush=True)

  # One full PPO update (rollout + learning) at the last batch size.
  cfg = ppo.PPOConfig(num_envs=args.batch[-1], rollout_len=args.rollout)
  init, update = ppo.make_train(env, cfg)
  ts = jax.jit(init)(jax.random.PRNGKey(0))
  update = jax.jit(update)
  ts, _ = update(ts, None)
  jax.block_until_ready(ts.actor)
  start = time.perf_counter()
  ts, _ = update(ts, None)
  jax.block_until_ready(ts.actor)
  print(f"PPO update (batch {cfg.num_envs}, rollout {cfg.rollout_len}): "
        f"{time.perf_counter() - start:.2f} s", flush=True)


if __name__ == "__main__":
  main()
