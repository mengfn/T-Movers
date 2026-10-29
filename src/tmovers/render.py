"""Matplotlib rendering of the puzzle: walls, load, carriers and pull forces."""

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation, patches  # noqa: E402

from tmovers import ants  # noqa: E402
from tmovers.env import TMoversEnv  # noqa: E402

_STATUS_STYLE = {
    ants.INFORMED: ("#e8a33d", "informed puller"),
    ants.PULLER: ("#2f6fd0", "puller"),
    ants.LIFTER: ("#8a8a8a", "lifter"),
}


def record(env: TMoversEnv, key: jax.Array, n_steps: int, every: int = 50):
  """Runs one puzzle and keeps (qpos, status, phi, time) every `every` steps."""

  def chunk(state, _):
    state = jax.lax.fori_loop(0, every, lambda _, s: env.step(s), state)
    return state, (state.data.qpos, state.status, state.phi, state.time,
                   state.solved)

  @jax.jit
  def run(key):
    return jax.lax.scan(chunk, env.reset(key), None, length=n_steps // every)

  _, frames = run(key)
  return jax.tree.map(np.asarray, frames)


def record_policy(env, policy, key: jax.Array, n_decisions: int):
  """Like `record`, for a ColonyEnv driven by `policy(state, key)`.

  One frame per decision; attached ants that do not pull show as lifters.
  """

  def step(carry, _):
    state, key = carry
    key, sub = jax.random.split(key)
    state = env.step(state, policy(state, sub))
    frame = (state.env.data.qpos, env.display_status(state), state.env.phi,
             state.env.time, state.env.solved)
    return (state, key), frame

  @jax.jit
  def run(key):
    return jax.lax.scan(step, (env.reset(key), key), None,
                        length=n_decisions)[1]

  return jax.tree.map(np.asarray, run(key))


class Renderer:

  def __init__(self, env: TMoversEnv, figsize=(8, 6)):
    self.env = env
    spec = env.spec
    self.fig, self.ax = plt.subplots(figsize=figsize)
    ax = self.ax
    ax.set_aspect("equal")
    x_end = spec.slit2 + spec.load_length
    ax.set_xlim(-1, x_end + 0.5)
    ax.set_ylim(-1, spec.breadth + 1)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    for (cx, cy), (hx, hy) in zip(np.asarray(env.wall_c),
                                  np.asarray(env.wall_h)):
      ax.add_patch(patches.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                     color="#555555"))
    # The middle chamber is too narrow for "2nd chamber" in the small arena.
    narrow = (spec.slit2 - spec.slit1) / (x_end + 1.5) < 0.2
    for x, label in [(spec.slit1 / 2, "1st"),
                     ((spec.slit1 + spec.slit2) / 2, "2nd"),
                     (spec.slit2 + spec.load_length / 2, "3rd")]:
      ax.text(x, spec.breadth + 0.4, label if narrow else f"{label} chamber",
              ha="center", fontsize=9, color="#555555")

    self.load = patches.Polygon(np.zeros((12, 2)), closed=True,
                                facecolor="#d6453d", edgecolor="#8c1c16")
    ax.add_patch(self.load)
    self.sites = ax.scatter([], [], s=10, zorder=3)
    # One arrow per site; non-pullers get zero length.
    zeros = np.zeros(env.n_sites)
    self.arrows = ax.quiver(zeros, zeros, zeros, zeros, angles="xy",
                            scale_units="xy", scale=1.0, width=0.003, zorder=4)
    self.title = ax.set_title("")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for c, l in _STATUS_STYLE.values()]
    # Below the plot: inside it, the legend covered the chamber labels.
    self.fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
                    frameon=False)
    self.fig.tight_layout(rect=(0, 0.06, 1, 1))
    self.fig.subplots_adjust(top=0.9)  # room for the per-frame title

  def draw(self, qpos, status, phi, time, solved=False):
    env = self.env
    frame = ants.SiteFrame(env.site_pos, env.site_normal, jnp.asarray(qpos),
                           jnp.zeros(3))
    self.load.set_xy(np.asarray(env.load_corners(jnp.asarray(qpos))))

    occupied = status != ants.EMPTY
    pos = np.asarray(frame.pos)
    colors = [_STATUS_STYLE[s][0] for s in status[occupied]]
    self.sites.set_offsets(pos[occupied] if occupied.any() else
                           np.zeros((0, 2)))
    self.sites.set_color(colors)

    pulling = (status == ants.INFORMED) | (status == ants.PULLER)
    dirs = np.asarray(ants.pull_dirs(frame.normal, jnp.asarray(phi)))
    length = 0.12 * env.spec.load_length * pulling
    self.arrows.set_offsets(pos)
    self.arrows.set_UVC(dirs[:, 0] * length, dirs[:, 1] * length)
    self.arrows.set_color([_STATUS_STYLE.get(s, ("none",))[0] for s in status])

    minutes, seconds = divmod(round(float(time), 1), 60)
    self.title.set_text(
        f"t = {int(minutes):02d}:{seconds:04.1f}   carriers {occupied.sum()}"
        f"   pullers {pulling.sum()}" + ("   SOLVED" if solved else ""))
    return self.load, self.sites, self.arrows, self.title

  def animate(self, frames, path, fps=20):
    qpos, status, phi, time, solved = frames
    anim = animation.FuncAnimation(
        self.fig, lambda i: self.draw(qpos[i], status[i], phi[i], time[i],
                                      solved[i]),
        frames=len(qpos), blit=False)
    writer = (animation.FFMpegWriter(fps=fps) if path.endswith(".mp4") else
              animation.PillowWriter(fps=fps))
    anim.save(path, writer=writer, dpi=90)
