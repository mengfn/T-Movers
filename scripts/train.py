"""Trains a shared ant policy with MAPPO.

  # GPU
  uv run --extra cuda12 python scripts/train.py --envs 1024 --updates 2000
  # quick CPU check
  uv run python scripts/train.py --envs 8 --rollout 16 --updates 4 --log-every 2
"""

import argparse
import json
import os
import time

import jax
from flax import serialization

from tmovers import evaluate, ppo
from tmovers.colony import ColonyConfig, ColonyEnv, paper_policy


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--envs", type=int, default=1024)
  parser.add_argument("--rollout", type=int, default=64)
  parser.add_argument("--updates", type=int, default=2000)
  parser.add_argument("--minibatches", type=int, default=8)
  parser.add_argument("--log-every", type=int, default=10)
  parser.add_argument("--controlled", default="uninformed",
                      choices=["uninformed", "all"])
  parser.add_argument("--extra-senses", default="",
                      help="comma-separated subset of walls,site,pose")
  parser.add_argument("--train-minutes", type=float, default=5.0,
                      help="episode time limit during training")
  parser.add_argument("--warm-start", action="store_true")
  parser.add_argument("--random-start", type=float, default=0.0)
  parser.add_argument("--bc-iters", type=int, default=300,
                      help="behaviour-cloning iterations on the paper's rule "
                      "before PPO (0 = start from scratch)")
  parser.add_argument("--local-coef", type=float, default=1.0)
  parser.add_argument("--prior-kl", type=float, default=0.0,
                      help="KL penalty towards the behaviour-cloned paper rule")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--eval-every", type=int, default=100,
                      help="updates between evaluations under the paper's "
                      "protocol (standard start, empty load)")
  parser.add_argument("--periodic-puzzles", type=int, default=64)
  parser.add_argument("--periodic-minutes", type=float, default=10.0)
  parser.add_argument("--hours", type=float, default=None,
                      help="stop training after this much wall-clock time")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out", default="runs/mappo")
  parser.add_argument("--eval-puzzles", type=int, default=64)
  parser.add_argument("--eval-minutes", type=float, default=30.0)
  args = parser.parse_args()

  senses = tuple(x for x in args.extra_senses.split(",") if x)
  train_env = ColonyEnv(ColonyConfig(
      controlled=args.controlled, extra_senses=senses,
      max_time=args.train_minutes * 60, warm_start=args.warm_start,
      random_start=args.random_start))
  # Evaluation follows the paper: initial pose, empty load, 30 min limit.
  env = ColonyEnv(ColonyConfig(controlled=args.controlled,
                               extra_senses=senses))
  cfg = ppo.PPOConfig(num_envs=args.envs, rollout_len=args.rollout,
                      updates=args.updates,
                      minibatches=min(args.minibatches, args.envs),
                      local_coef=args.local_coef,
                      prior_kl_coef=args.prior_kl, lr=args.lr, seed=args.seed)
  os.makedirs(args.out, exist_ok=True)
  print(f"devices {jax.devices()} | obs {env.obs_dim} | agents {env.n_agents}")
  start = time.perf_counter()
  log = open(os.path.join(args.out, "log.jsonl"), "w")

  # Periodic evaluation under the paper's protocol, against the paper's rule.
  eval_keys = jax.random.split(jax.random.PRNGKey(20_000),
                               args.periodic_puzzles)
  n_eval = int(args.periodic_minutes * 60 / env.config.decision_dt)
  actor_net = ppo.Actor(cfg.hidden)
  evaluator = evaluate.make_evaluator(
      env, lambda p, s, k: ppo.sample(actor_net.apply(p, s.obs), k), eval_keys,
      n_eval)
  paper_eval = evaluate.make_evaluator(
      env, lambda p, s, k: paper_policy(env, s, k), eval_keys, n_eval)(None)
  summary = lambda m: (f"solved {m['solved'].mean():5.1%} | final dist "
                       f"{m['final_distance'].mean():.2f} d_cor")
  print(f"eval paper rule: {summary(paper_eval)}", flush=True)
  eval_log = open(os.path.join(args.out, "eval.jsonl"), "w")

  def callback(row, ts):
    if row["update"] % args.eval_every == 0:
      m = evaluator(ts.actor)
      print(f"eval update {row['update']:5d}: {summary(m)}",
            flush=True)
      eval_log.write(json.dumps({
          "update": row["update"], "solved": float(m["solved"].mean()),
          "final_distance": float(m["final_distance"].mean())}) + "\n")
      eval_log.flush()
    elapsed = time.perf_counter() - start
    steps = row["update"] * cfg.num_envs * cfg.rollout_len
    print(f"update {row['update']:5d} | {elapsed:6.0f}s | "
          f"{steps / elapsed:8.0f} decisions/s | reward {row['reward']:+.4f} | "
          f"solved {int(row['solved'])}/{int(row['episodes'])} | "
          f"dist {row['distance']:.2f} d_cor | pull {row['pull_frac']:.2f} | "
          f"entropy {row['entropy']:.3f} | v_loss {row['v_loss']:.4f}",
          flush=True)
    log.write(json.dumps(row) + "\n")
    log.flush()
    with open(os.path.join(args.out, "actor.msgpack"), "wb") as f:
      f.write(serialization.to_bytes(ts.actor))

  actor_params = None
  if args.bc_iters:
    actor_params, bc = ppo.behavior_cloning(
        train_env, lambda s, k: paper_policy(train_env, s, k), cfg,
        iters=args.bc_iters, seed=args.seed)
    print(f"behaviour cloning: nll {bc[0][0]:.3f} -> {bc[-1][0]:.3f}, "
          f"pull agreement {bc[-1][1]:.3f}", flush=True)
    with open(os.path.join(args.out, "actor_bc.msgpack"), "wb") as f:
      f.write(serialization.to_bytes(actor_params))
    print(f"eval BC start: {summary(evaluator(actor_params))}",
          flush=True)
  budget = args.hours * 3600 if args.hours else None
  prior = actor_params if args.prior_kl > 0 else None
  ts, _ = ppo.train(train_env, cfg, args.log_every, callback, budget,
                    actor_params, prior)

  if args.eval_puzzles:
    n = int(args.eval_minutes * 60 / (env.config.action_repeat * env.config.dt))
    keys = jax.random.split(jax.random.PRNGKey(10_000), args.eval_puzzles)
    m = evaluate.evaluate_policy(env, ppo.make_policy(ts.actor, cfg.hidden,
                                                      deterministic=False),
                                 keys, n)
    print(evaluate.report("MAPPO", m))
    if actor_params is not None:
      m = evaluate.evaluate_policy(
          env, ppo.make_policy(actor_params, cfg.hidden, deterministic=False),
          keys, n)
      print(evaluate.report("BC (start)", m))
    m = evaluate.evaluate_policy(env, lambda s, k: paper_policy(env, s, k),
                                 keys, n)
    print(evaluate.report("paper rule", m))


if __name__ == "__main__":
  main()
