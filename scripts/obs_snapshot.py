"""Snapshot of one large-arena attempt (paper rule) for the observation figure.

  uv run python scripts/obs_snapshot.py 900 runs/obs_snapshot.json

Runs PRNGKey(7) for N decisions (900 = 90 s), then prints and saves every
attached ant's role, position, normal and 8-number observation.
"""
import json, sys
import jax, jax.numpy as jnp, numpy as np
from tmovers import ants
from tmovers.colony import ColonyEnv, paper_policy

env = ColonyEnv()
T = int(sys.argv[1]) if len(sys.argv) > 1 else 900  # decisions (0.1 s each)

@jax.jit
def run(key):
  def step(carry, _):
    s, k = carry
    k, sub = jax.random.split(k)
    return (env.step(s, paper_policy(env, s, sub)), k), None
  return jax.lax.scan(step, (env.reset(key), key), None, length=T)[0][0]

s = run(jax.random.PRNGKey(7))
base = env.base
frame = base.site_frame(s.env.data)
status = np.asarray(env.display_status(s))
obs = np.asarray(s.obs)  # masked: zero for informed ants (not controlled)
obs_all = np.asarray(env._observe(s.env, s.pull))  # every attached ant
n = np.asarray(frame.normal); pos = np.asarray(frame.pos)
phi = np.asarray(s.env.phi)
p_dirs = np.asarray(ants.pull_dirs(frame.normal, s.env.phi))
names = {ants.INFORMED: "informed", ants.PULLER: "puller", ants.LIFTER: "lifter"}
att = [i for i in range(env.n_agents) if status[i] != ants.EMPTY]
print("t =", float(s.env.time), "carriers", len(att), "pose", np.asarray(s.env.data.qpos).round(3))
for i in att:
  o = obs_all[i]
  print(i, names[int(status[i])], "obs", o.round(3), "pos", pos[i].round(2))
out = {
  "time": float(s.env.time), "qpos": np.asarray(s.env.data.qpos).tolist(),
  "outline": np.asarray(base.load_corners(s.env.data.qpos)).tolist(),
  "sites": [{"i": i, "role": names[int(status[i])], "x": float(pos[i, 0]),
             "y": float(pos[i, 1]), "nx": float(n[i, 0]), "ny": float(n[i, 1]),
             "obs": obs_all[i].tolist(), "controlled": bool(np.asarray(s.mask)[i])}
            for i in att],
}
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/snap.json", "w"))
