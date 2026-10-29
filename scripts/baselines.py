"""Compares reference policies with the paper's rule-based ant model.

  uv run python scripts/baselines.py --puzzles 64 --minutes 30
"""

import argparse

import jax

from tmovers import evaluate
from tmovers.colony import (ColonyEnv, greedy_policy, paper_policy,
                            random_policy)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--puzzles", type=int, default=16)
  parser.add_argument("--minutes", type=float, default=15.0)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  env = ColonyEnv()
  n = int(args.minutes * 60 / (env.config.action_repeat * env.config.dt))
  keys = jax.random.split(jax.random.PRNGKey(args.seed), args.puzzles)
  print(f"{args.puzzles} puzzles, {args.minutes} min each, optimal path "
        f"{env.d_start / env.cspace.d_cor:.2f} d_cor")
  print(evaluate.report("paper model", evaluate.evaluate_paper_model(env, keys, n)))
  for name, policy in [("paper policy", paper_policy),
                       ("greedy", greedy_policy), ("random", random_policy)]:
    m = evaluate.evaluate_policy(env, lambda s, k: policy(env, s, k), keys, n)
    print(evaluate.report(name, m))


if __name__ == "__main__":
  main()
