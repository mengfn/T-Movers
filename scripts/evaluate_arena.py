"""Evaluates the paper's model, the paper rule and saved actors on one arena.

Same protocol and seeds as train.py's final evaluation (standard start, empty
load, PRNGKey(10_000)), so results on different arenas are comparable:

  uv run python scripts/evaluate_arena.py --arena small --puzzles 64 \
      --minutes 30 --actor MAPPO=runs/delta/run_B/actor.json \
      --actor BC=runs/delta/run_B/actor_bc.json --out runs/small_eval.json

Actors trained on one arena are evaluated zero-shot on another: the ants'
observations do not depend on the arena's size.
"""

import argparse
import json
import time

import jax
import numpy as np
from flax import serialization

from tmovers import evaluate, ppo
from tmovers.colony import ColonyConfig, ColonyEnv, paper_policy


def load_actor(path, obs_dim, hidden=128):
  template = ppo.Actor(hidden).init(jax.random.PRNGKey(0),
                                    np.zeros((1, obs_dim)))
  if path.endswith(".json"):
    # JSON checkpoints survive text-only file transfers (see delta notes).
    to_arrays = lambda d: ({k: to_arrays(v) for k, v in d.items()}
                           if isinstance(d, dict) else np.asarray(d, np.float32))
    with open(path) as f:
      return serialization.from_state_dict(template, to_arrays(json.load(f)))
  with open(path, "rb") as f:
    return serialization.from_bytes(template, f.read())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--arena", default="small")
  parser.add_argument("--n-sites", type=int, default=None,
                      help="override the arena's N_max (attachment sites)")
  parser.add_argument("--puzzles", type=int, default=64)
  parser.add_argument("--minutes", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=10_000)
  parser.add_argument("--actor", action="append", default=[],
                      help="NAME=path to an actor checkpoint (.msgpack/.json)")
  parser.add_argument("--no-model", action="store_true",
                      help="skip the paper's model (same rules as the paper "
                      "rule; see validate_interface.py)")
  parser.add_argument("--out", required=True)
  args = parser.parse_args()

  env = ColonyEnv(ColonyConfig(arena=args.arena, n_sites=args.n_sites))
  n = int(args.minutes * 60 / env.config.decision_dt)
  keys = jax.random.split(jax.random.PRNGKey(args.seed), args.puzzles)
  print(f"{args.arena} arena, N_max {env.n_agents} | {args.puzzles} puzzles "
        f"x {args.minutes} min | optimal path {env.d_start / env.cspace.d_cor:.2f} d_cor | "
        f"devices {jax.devices()}", flush=True)

  runs = [] if args.no_model else [
      ("paper model", lambda: evaluate.evaluate_paper_model(env, keys, n))]
  runs.append(("paper rule", lambda: evaluate.evaluate_policy(
      env, lambda s, k: paper_policy(env, s, k), keys, n)))
  for spec in args.actor:
    name, path = spec.split("=", 1)
    params = load_actor(path, env.obs_dim)
    runs.append((name, lambda p=params: evaluate.evaluate_policy(
        env, ppo.make_policy(p, deterministic=False), keys, n)))

  results = {}
  for name, run in runs:
    start = time.time()
    m = run()
    print(f"{evaluate.report(name, m)} | {time.time() - start:.0f} s",
          flush=True)
    results[name] = {k: np.asarray(v).tolist() for k, v in m.items()}
    with open(args.out, "w") as f:
      json.dump({"arena": args.arena, "n_sites": env.n_agents,
                 "puzzles": args.puzzles,
                 "minutes": args.minutes, "seed": args.seed,
                 "results": results}, f)


if __name__ == "__main__":
  main()
