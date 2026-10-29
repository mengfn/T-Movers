"""Standalone three.js viewer for recorded trajectories, in the style of
brax.io.html.

  frames = tmovers.render.record(env, key, n_steps, every=100)
  tmovers.html.save("media/attempt.html", env, frames)

The page loads three.js from a CDN and embeds the trajectory, so it opens
from disk in any browser.
"""

import importlib.resources
import json

import numpy as np

from tmovers import geometry, model
from tmovers.env import TMoversEnv


def render(env: TMoversEnv, frames, title: str = "T-Movers",
           fps: float = 20.0) -> str:
  """Returns the viewer page for frames from `tmovers.render.record`.

  Args:
    env: the environment the frames were recorded from.
    frames: (qpos, status, phi, time, solved) arrays with a leading time axis.
    title: heading shown on the page.
    fps: playback frames per second at 1x.
  """
  qpos, status, phi, time, solved = (np.asarray(f) for f in frames)
  spec = env.spec

  load_c, load_h, _ = geometry.load_boxes(spec)
  wall_c, wall_h = np.asarray(env.wall_c), np.asarray(env.wall_h)
  corners = np.concatenate([wall_c - wall_h, wall_c + wall_h])
  rnd = lambda a, d=4: np.round(np.asarray(a, float), d).tolist()

  data = {
      "title": title,
      "fps": fps,
      "bounds": rnd([*corners.min(0), *corners.max(0)]),
      "wall_z": model._Z_HALF_WALL,
      "load_z": model._Z_HALF_LOAD,
      "walls": [[rnd(c), rnd(h)] for c, h in zip(wall_c, wall_h)],
      "load": [[rnd(c), rnd(h)] for c, h in zip(load_c, load_h)],
      "sites": {"pos": rnd(env.site_pos), "normal": rnd(env.site_normal)},
      "ant_radius": round(0.4 * spec.load_width, 4),
      "arrow_length": round(0.12 * spec.load_length, 4),
      "qpos": rnd(qpos),
      # One digit per site and frame; pull angles in whole degrees.
      "status": ["".join(map(str, s)) for s in status.astype(int)],
      "phi": np.round(np.degrees(phi)).astype(int).tolist(),
      "time": rnd(time, 2),
      "solved": solved.astype(bool).tolist(),
  }
  template = (importlib.resources.files("tmovers") /
              "viewer.html").read_text()
  payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
  return template.replace("__TMOVERS_DATA__", payload)


def save(path: str, env: TMoversEnv, frames, **kwargs) -> None:
  with open(path, "w") as f:
    f.write(render(env, frames, **kwargs))
