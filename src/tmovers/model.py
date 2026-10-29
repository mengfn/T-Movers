"""MJCF model of the puzzle: a planar rigid T-load inside static walls.

The load has three joints (slide x, slide y, hinge z), so it moves only in the
plane and qpos is exactly its world pose (x, y, theta) at the centre of mass.
Gravity is off; the floor and the transparent cover of the real arena are
represented by the planar constraint rather than by geoms.
"""

import mujoco

from tmovers import geometry

_Z_HALF_LOAD = 0.25
_Z_HALF_WALL = 0.5


def build_xml(spec: geometry.PuzzleSpec, timestep: float = 0.01) -> str:
  load_c, load_h, _ = geometry.load_boxes(spec)
  wall_c, wall_h = geometry.wall_boxes(spec)

  load_geoms = "\n".join(
      f'      <geom type="box" pos="{c[0]:.6f} {c[1]:.6f} 0" '
      f'size="{h[0]:.6f} {h[1]:.6f} {_Z_HALF_LOAD}" rgba="0.85 0.15 0.15 1"/>'
      for c, h in zip(load_c, load_h))
  wall_geoms = "\n".join(
      f'    <geom type="box" pos="{c[0]:.6f} {c[1]:.6f} 0" '
      f'size="{h[0]:.6f} {h[1]:.6f} {_Z_HALF_WALL}" rgba="0.5 0.5 0.5 1"/>'
      for c, h in zip(wall_c, wall_h))

  # The load's mass scales linearly with N_max, as in the paper (SI Note 1),
  # rather than with its area: the large arena keeps density 1, the small one
  # (1/8 the size, 1/8 the sites) gets density 8. In the overdamped regime the
  # mass does not change the load's speed, but the contact solver needs it to
  # stay comparable to damping * dt: with area-scaled mass the small load
  # chattered at ~30% of its free speed and sank 14% of its width into walls.
  large = geometry.SPECS["large"]
  density = ((large.load_length / spec.load_length)**2 * spec.n_sites /
             large.n_sites)

  # Euler integrates joint damping implicitly, which keeps the strongly
  # overdamped load stable for any damping value. With MuJoCo's default
  # solimp, MJX's box-box test intermittently drops the resting wall contact
  # (equilibrium depth ~1e-5 cm), kicking the load off the wall every ~25
  # steps; the softer solimp keeps the contact resolved (depth <= 2e-4 cm).
  return f"""
<mujoco model="t_movers">
  <option timestep="{timestep}" gravity="0 0 0" integrator="Euler"
          iterations="8" ls_iterations="8"/>
  <default>
    <geom friction="0.2 0.005 0.0001" solref="0.02 1" solimp="0.5 0.9 0.01"
          density="{density:.6f}"/>
  </default>
  <worldbody>
{wall_geoms}
    <body name="load">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="theta" type="hinge" axis="0 0 1"/>
{load_geoms}
    </body>
  </worldbody>
</mujoco>
"""


def build_model(spec: geometry.PuzzleSpec,
                timestep: float = 0.01) -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(build_xml(spec, timestep))
