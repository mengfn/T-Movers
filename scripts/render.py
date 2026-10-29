"""Renders one simulated puzzle attempt.

  uv run python scripts/render.py --minutes 10 --out media/attempt.gif
  uv run python scripts/render.py --minutes 10 --out media/attempt.html
  uv run python scripts/render.py --policy greedy --out media/greedy.html
  uv run python scripts/render.py --policy runs/mappo/actor.msgpack \
      --out media/mappo.html

--policy is "paper" (the paper's rule-based ants, default), "greedy",
"random", or the path to an actor checkpoint saved by scripts/train.py.
A .html output is an interactive three.js viewer; anything else is an
animation written by matplotlib.
"""

import argparse
import os

import jax
import numpy as np

from tmovers import html, ppo
from tmovers.colony import ColonyEnv, greedy_policy, random_policy
from tmovers.env import TMoversEnv
from tmovers.render import Renderer, record, record_policy


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--arena", default="large", choices=["large", "small"])
  parser.add_argument("--policy", default="paper")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--minutes", type=float, default=10.0)
  parser.add_argument("--every", type=int, default=100,
                      help="physics steps per frame for --policy paper "
                      "(dt = 0.01 s); colony policies use one frame per "
                      "decision")
  parser.add_argument("--out", default="media/attempt.gif")
  parser.add_argument("--snapshots", type=int, default=4,
                      help="also save this many still frames as PNG")
  args = parser.parse_args()

  key = jax.random.PRNGKey(args.seed)
  if args.policy == "paper":
    env = TMoversEnv(args.arena)
    n_steps = int(args.minutes * 60 / env.dt)
    frames = record(env, key, n_steps, args.every)
  else:
    colony = ColonyEnv()
    env = colony.base
    if args.policy == "greedy":
      policy = lambda s, k: greedy_policy(colony, s, k)
    elif args.policy == "random":
      policy = lambda s, k: random_policy(colony, s, k)
    else:
      from flax import serialization
      template = ppo.Actor(128).init(jax.random.PRNGKey(0),
                                     np.zeros((1, colony.obs_dim)))
      with open(args.policy, "rb") as f:
        params = serialization.from_bytes(template, f.read())
      policy = ppo.make_policy(params)
    decision = colony.config.action_repeat * colony.config.dt
    frames = record_policy(colony, policy, key,
                           int(args.minutes * 60 / decision))
  solved = frames[4]
  if solved.any():  # stop shortly after the puzzle is solved
    end = min(len(solved), int(solved.argmax()) + 20)
    frames = tuple(f[:end] for f in frames)

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  if args.out.endswith(".html"):
    title = f"T-Movers · {args.arena} ants · {args.policy} · seed {args.seed}"
    html.save(args.out, env, frames, title=title)
    print(f"{len(frames[0])} frames, solved: {bool(solved.any())} -> {args.out}")
    return
  renderer = Renderer(env)
  renderer.animate(frames, args.out)
  stem = os.path.splitext(args.out)[0]
  for k, i in enumerate(np.linspace(0, len(frames[0]) - 1, args.snapshots,
                                    dtype=int)):
    renderer.draw(*(f[i] for f in frames))
    renderer.fig.savefig(f"{stem}_{k}.png", dpi=110)
  print(f"{len(frames[0])} frames, solved: {bool(solved.any())} -> {args.out}")


if __name__ == "__main__":
  main()
