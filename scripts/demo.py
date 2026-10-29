"""Records demo attempts of a colony policy and renders the fastest solve.

  uv run python scripts/demo.py --policy paper --out media/demo_paper
  uv run python scripts/demo.py --policy runs/delta/run_B/actor.msgpack \
      --out media/demo_mappo
  uv run python scripts/demo.py --arena small --minutes 15 --out media/small_paper

Runs --seeds attempts in parallel under the paper's protocol (initial pose,
empty load), picks the fastest solved one (or the closest to the goal), and
writes <out>.html (three.js viewer), <out>.gif (animation) and <out>.png
(four still frames).
"""

import argparse
import json
import os

import jax
import numpy as np
from flax import serialization

from tmovers import html, ppo
from tmovers.colony import ColonyConfig, ColonyEnv, paper_policy
from tmovers.render import Renderer


def record_batch(env, policy, keys, n_decisions, every):
  """Frames (qpos, status, phi, time, solved), shape (T, B, ...)."""
  batch = keys.shape[0]

  def step(carry, _):
    state, key = carry
    key, sub = jax.random.split(key)
    action = jax.vmap(policy)(state, jax.random.split(sub, batch))
    return (jax.vmap(env.step)(state, action), key), None

  def chunk(carry, _):
    carry, _ = jax.lax.scan(step, carry, None, length=every)
    s = carry[0]
    return carry, (s.env.data.qpos, jax.vmap(env.display_status)(s),
                   s.env.phi, s.env.time, s.env.solved)

  @jax.jit
  def run(keys):
    state = jax.vmap(env.reset)(keys)
    return jax.lax.scan(chunk, (state, keys[0]), None,
                        length=n_decisions // every)[1]

  return jax.tree.map(np.asarray, run(keys))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--policy", default="paper")
  parser.add_argument("--arena", default="large")
  parser.add_argument("--n-sites", type=int, default=None,
                      help="override the arena's N_max (attachment sites)")
  parser.add_argument("--seeds", type=int, default=8)
  parser.add_argument("--minutes", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--out", required=True, help="output path without suffix")
  args = parser.parse_args()

  env = ColonyEnv(ColonyConfig(arena=args.arena, n_sites=args.n_sites))
  if args.policy == "paper":
    policy, name = (lambda s, k: paper_policy(env, s, k)), "paper rule"
  else:
    template = ppo.Actor(128).init(jax.random.PRNGKey(0),
                                   np.zeros((1, env.obs_dim)))
    if args.policy.endswith(".json"):
      to_arrays = lambda d: ({k: to_arrays(v) for k, v in d.items()}
                             if isinstance(d, dict) else np.asarray(d, np.float32))
      with open(args.policy) as f:
        params = serialization.from_state_dict(template,
                                               to_arrays(json.load(f)))
    else:
      with open(args.policy, "rb") as f:
        params = serialization.from_bytes(template, f.read())
    policy, name = ppo.make_policy(params, deterministic=False), "MAPPO"

  every = int(round(1.0 / env.config.decision_dt))  # one frame per second
  n = int(args.minutes * 60 / env.config.decision_dt)
  keys = jax.random.split(jax.random.PRNGKey(args.seed), args.seeds)
  qpos, status, phi, time, solved = record_batch(env, policy, keys, n, every)

  solved_any = solved.any(0)
  first = np.where(solved_any, solved.argmax(0), len(solved))
  dist = np.asarray(jax.vmap(jax.vmap(env.cspace.distance))(qpos[-1:]))[0]
  pick = int(np.argmin(first)) if solved_any.any() else int(np.argmin(dist))
  end = min(len(solved), int(first[pick]) + 10)
  frames = tuple(f[:end, pick] for f in (qpos, status, phi, time, solved))
  summary = {
      "policy": name,
      "arena": args.arena,
      "n_sites": env.n_agents,
      "seeds": args.seeds,
      "minutes": args.minutes,
      "solved": int(solved_any.sum()),
      "solve_times_s": [int(t) + 1 for t in first[solved_any]],
      "shown_seed_index": pick,
      "shown_solved": bool(solved_any[pick]),
      "shown_time_s": float(frames[3][-1]),
  }
  print(json.dumps(summary))

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out + ".json", "w") as f:
    json.dump(summary, f, indent=2)
  title = f"T-Movers · {args.arena} arena, N_max {env.n_agents} · {name} · " + (
      f"solved in {int(first[pick]) + 1} s" if solved_any[pick] else "best attempt")
  html.save(args.out + ".html", env.base, frames, title=title)

  renderer = Renderer(env.base, figsize=(7, 5))
  gif_frames = tuple(f[::4] for f in frames)  # one frame per 4 s
  renderer.animate(gif_frames, args.out + ".gif", fps=12)
  # The last still is the moment of solving: the clip runs 10 s past it, by
  # which time the load may have left the open end of the third chamber.
  last = int(first[pick]) if solved_any[pick] else len(frames[0]) - 1
  stills = np.linspace(0, last, 4, dtype=int)
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
  for ax, i in zip(axes, stills):
    renderer.draw(*(f[i] for f in frames))
    renderer.fig.canvas.draw()
    ax.imshow(np.asarray(renderer.fig.canvas.buffer_rgba()))
    ax.axis("off")
  fig.suptitle(f"{name}, {args.arena} arena (N_max {env.n_agents}): "
               f"{title.split('· ')[-1]}", fontsize=13)
  fig.tight_layout()
  fig.savefig(args.out + ".png", dpi=110)


if __name__ == "__main__":
  main()
