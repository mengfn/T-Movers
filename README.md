# T-Movers

GPU-parallel MJX simulator of the ant "piano-movers" T-load puzzle from
Dreyer, Haluts, Korman, Gov, Fonio & Feinerman, *Comparing cooperative geometric
puzzle solving in ants versus humans*, PNAS 2024
([doi:10.1073/pnas.2414274121](https://doi.org/10.1073/pnas.2414274121)).

## How it maps to the paper

| Paper (SI Appendix Note 1) | Here |
|---|---|
| Box2D 2D rigid body | MJX body with slide-x, slide-y, hinge-z joints, so `qpos = (x, y, θ)` |
| Arena and load sizes (Table S2, "with skirt") | `geometry.SPECS["large"]` (N_max = 120) and `["small"]` (N_max = 15) |
| Ants as force vectors at attachment sites | `ants.py`: 120 sites, status empty / informed / puller / lifter |
| Gillespie event loop | fixed-step events with probability `1 - exp(-k dt)` (vectorizes under `vmap`) |
| Box2D `linearDamping` / `angularDamping` per carrier (Eq. 3) | implicit joint damping `dof_damping`, updated every step |
| Parameters (Table S5) | `ants.AntParams` defaults |

## Usage

```python
import jax
from tmovers import TMoversEnv, rollout

env = TMoversEnv("large")
keys = jax.random.split(jax.random.PRNGKey(0), 4096)

# Paper's rule-based ants, many puzzles in parallel:
state, metrics = rollout(env, keys, n_steps=60_000, record_every=100)

# Or drive it yourself (e.g. from a policy):
state = jax.vmap(env.reset)(keys)
step = jax.jit(jax.vmap(env.step))
state = step(state)                    # rule-based ants
state = step(state, pull_angles)       # (batch, 120) in [-1, 1] overrides puller directions
```

`EnvState.solved` turns true once the whole load is inside the 3rd chamber.
`env.metrics(state)` returns the pose, velocity, carrier counts and the order
parameter M.

## Ant colony for RL (`ColonyEnv`)

Each of the 120 attachment sites is an agent slot, and **one shared policy
controls every attached ant**. Ants still join and leave the load following
the paper's rates (k_on, k_forget, k_off, and letting go next to walls), but
while attached, each ant chooses its own behaviour:

| | |
|---|---|
| **Action** (per ant, every 0.2 s) | `pull > 0.5` → pull with f0, otherwise lift; `angle ∈ [-1, 1]` → direction as a fraction of φ_max = 52° from the site normal |
| **Observation** (per ant, 13 numbers, in the ant's own frame) | attached / informed / knows-target flags, local load velocity (the paper's mechanical cue), direction to the next exit (only for ants that know it), wall distance at and ahead of the site, position on the load, previous action |
| **Mask** | which sites hold an ant (empty sites have all-zero observations) |
| **Global state** (critic only) | pose, puzzle state (a…h), velocity, number of carriers, distance to goal, time |
| **Reward** (team) | progress along the shortest collision-free path to the goal, in d_cor, plus 10 on solving |

`ColonyConfig(target_info=...)` sets who knows the way: `"informed"` (the
paper's newly attached ants, ~11 s), `"all"` or `"none"`.
`global_obs=True` adds the load's pose and puzzle state to every ant's
observation.

Configuration space (`configspace.py`): a collision-checked (x, y, θ) grid with
the shortest-path distance to the goal, using the paper's metric
ds² = dx² + dy² + (r_av dθ)². The optimum from the start is 7.9 d_cor and
passes through the paper's optimal states a → c → e′ → f → h. It is built in
about 5 s and cached in `~/.cache/tmovers`. `classify()` maps poses to the
puzzle states of Table S3 (best-effort: the table doesn't define g, so g
means "none of the others").

```bash
uv run python scripts/baselines.py --puzzles 64 --minutes 30     # paper model vs greedy vs random
uv run --extra cuda12 python scripts/train.py --envs 1024 --updates 2000
uv run python scripts/render.py --policy runs/mappo/actor.msgpack --out media/mappo.html
```

`scripts/train.py` runs MAPPO: a shared actor network for all ants and a
centralized critic, with the loss averaged over attached ants only. It saves
`actor.msgpack` and `log.jsonl` under `--out`, then evaluates on the paper's
metrics (solve rate, time to solve, path length in d_cor).

## Visualisation

```bash
uv run python scripts/render.py --minutes 15 --out media/attempt.html   # three.js viewer
uv run python scripts/render.py --minutes 15 --out media/attempt.gif    # matplotlib animation
```

The `.html` output is a single standalone page, in the style of `brax.io.html`.
It embeds the trajectory and loads three.js from jsDelivr. It has play/pause, a
timeline scrubber, speed control, and a top or orbiting 3D camera. It shows the
ants by role, their pull directions, and the trail of the centre of mass. From
Python:

```python
from tmovers import html
from tmovers.render import record

frames = record(env, key, n_steps=90_000, every=50)   # one puzzle, every 0.5 s
html.save("media/attempt.html", env, frames)
```

Pages over about 1 MB may not open as `file://` in some embedded previews.
`python3 -m http.server --directory media` serves them over HTTP instead.

## Setup

```bash
uv sync                  # CPU (macOS / Linux)
uv sync --extra cuda12   # NVIDIA GPU (Linux); or cuda13 with driver >= 580
uv run pytest
uv run python scripts/benchmark.py --batch 1024 4096
```

## Modelling notes

- Units are cm and seconds. Forces are in the paper's units (f0 = 2.8).
- The paper calibrates damping as `γ̄·m_T`, so the load's mass drops out of the
  overdamped dynamics. The terminal velocity is F / (1.48 f0 N_eff).
- `dt = 0.01 s` is used for both the physics and the ant events. Event rates
  are ≤ ~10/s, so `k·dt ≪ 1`.
- The contacts use `solimp="0.5 0.9 0.01"`. With MuJoCo's default `solimp`,
  MJX's box-box test keeps dropping the contact of a load resting on a wall.
  The load then gets kicked off the wall, which corrupts the local-velocity
  signal the ants respond to (see `model.py`).
- Not implemented yet: the human solvers.
