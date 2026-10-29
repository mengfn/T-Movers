"""Vectorized carrier-ant model (SI Appendix Note 1 and Table S5).

The paper runs a Gillespie algorithm, firing one event at a time. On a GPU we
instead advance all attachment sites together with a fixed step dt: each
event with rate k fires with probability 1 - exp(-k dt), and competing events
of one site are resolved by their relative rates. This converges to the same
dynamics for k dt << 1 (largest rates here are ~10/s, so use dt <= 0.01 s).

Site status codes: EMPTY, INFORMED (informed puller), PULLER (uninformed
puller), LIFTER (uninformed, applies no force).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

EMPTY, INFORMED, PULLER, LIFTER = 0, 1, 2, 3


@dataclass(frozen=True)
class AntParams:
  k_on: float = 0.015  # attachment rate per empty site [1/s]
  k_forget: float = 0.09  # informed -> uninformed [1/s]
  k_c: float = 1.0  # basal role-switching rate [1/s]
  k_off: float = 0.015  # detachment rate per uninformed carrier [1/s]
  k_orient: float = 0.7  # reorientation rate per puller [1/s]
  f0: float = 2.8  # pulling force of one puller
  f_ind: float = 10.0  # individuality parameter [f0]
  gamma: float = 1.48  # linear damping per carrier [f0 s / cm]
  gamma_rot: float = 1.44  # angular damping per carrier [f0 s]
  phi_max_deg: float = 52.0  # max pull angle away from the outward normal
  wall_eps: float = 0.05  # carriers this close to a wall detach [cm]
  n0_fraction: float = 0.2  # N0 = N_max / 5 (SI code listing)


class SiteFrame:
  """World-frame positions, normals and velocities of the attachment sites."""

  def __init__(self, site_pos, site_normal, qpos, qvel):
    c, s = jnp.cos(qpos[2]), jnp.sin(qpos[2])
    rot = jnp.array([[c, -s], [s, c]])
    self.r = site_pos @ rot.T  # COM -> site, world frame
    self.pos = qpos[:2] + self.r
    self.normal = site_normal @ rot.T
    omega = qvel[2]
    self.vel = qvel[:2] + omega * jnp.stack([-self.r[:, 1], self.r[:, 0]], -1)


def pull_dirs(normal, phi):
  """Pull direction p_i: the outward normal rotated by phi_i."""
  c, s = jnp.cos(phi), jnp.sin(phi)
  return jnp.stack([c * normal[:, 0] - s * normal[:, 1],
                    s * normal[:, 0] + c * normal[:, 1]], -1)


def clipped_angle(normal, direction, phi_max):
  """Angle from the normal to `direction`, limited to +-phi_max."""
  cross = normal[:, 0] * direction[:, 1] - normal[:, 1] * direction[:, 0]
  dot = jnp.sum(normal * direction, -1)
  return jnp.clip(jnp.arctan2(cross, dot), -phi_max, phi_max)


def effective_carriers(n_att, n_max, p: AntParams):
  """Eq. (3): below N0 carriers the load behaves as if fully occupied."""
  return jnp.where(n_att <= p.n0_fraction * n_max, n_max, n_att)


def damping(n_att, n_max, inertia_ratio, p: AntParams):
  """Joint damping [d_x, d_y, d_theta] for the current number of carriers.

  Box2D damps v by m * c * v with c = gamma_bar * N_eff, and the paper
  calibrates gamma_bar * m_T, so the force damping is gamma * f0 * N_eff.
  The torque damping is I * c_rot = gamma_rot * f0 * N_eff * (I / m_T).
  """
  n_eff = effective_carriers(n_att, n_max, p)
  d_lin = p.gamma * p.f0 * n_eff
  d_rot = p.gamma_rot * p.f0 * n_eff * inertia_ratio
  return jnp.stack([d_lin, d_lin, d_rot]), d_lin


def informed_targets(pos, spec):
  """Direction each informed puller aims for, from the chamber of its site."""
  mid_y = spec.breadth / 2
  to1 = jnp.stack([spec.slit1 - pos[:, 0], mid_y - pos[:, 1]], -1)
  to2 = jnp.stack([spec.slit2 - pos[:, 0], mid_y - pos[:, 1]], -1)
  plus_x = jnp.broadcast_to(jnp.array([1.0, 0.0]), pos.shape)
  x = pos[:, :1]
  return jnp.where(x < spec.slit1, to1, jnp.where(x < spec.slit2, to2, plus_x))


def update(key, status, phi, frame: SiteFrame, near_wall, d_lin, dt, spec,
           p: AntParams):
  """Advances the carriers by one step of length dt.

  Returns the new (status, phi).
  """
  phi_max = jnp.deg2rad(p.phi_max_deg)
  u = jax.random.uniform(key, (4,) + status.shape)
  prob = lambda rate: 1.0 - jnp.exp(-rate * dt)

  p_dir = pull_dirs(frame.normal, phi)
  f_loc = d_lin * frame.vel  # Eq. (2)
  signal = jnp.sum(p_dir * f_loc, -1) / (p.f_ind * p.f0)
  r_lp = p.k_c * jnp.exp(signal)  # lifter -> puller, Eq. (1)
  r_pl = p.k_c * jnp.exp(-signal)  # puller -> lifter

  # Carriers that get too close to a wall let go.
  status = jnp.where(near_wall, EMPTY, status)

  # Informed carriers forget and pick a role by Eq. (14).
  forget = (status == INFORMED) & (u[0] < prob(p.k_forget))
  becomes_puller = u[1] < jax.nn.sigmoid(2.0 * signal)
  forgot_role = jnp.where(becomes_puller, PULLER, LIFTER)

  # Uninformed carriers either detach or switch roles (competing events).
  r_switch = jnp.where(status == PULLER, r_pl, r_lp)
  r_total = p.k_off + r_switch
  uninformed = (status == PULLER) | (status == LIFTER)
  fires = uninformed & (u[0] < prob(r_total))
  detach = fires & (u[1] * r_total < p.k_off)
  switch = fires & ~detach
  switched_role = jnp.where(status == PULLER, LIFTER, PULLER)

  attach = (status == EMPTY) & ~near_wall & (u[2] < prob(p.k_on))

  new_status = status
  new_status = jnp.where(forget, forgot_role, new_status)
  new_status = jnp.where(detach, EMPTY, new_status)
  new_status = jnp.where(switch, switched_role, new_status)
  new_status = jnp.where(attach, INFORMED, new_status)

  # Pullers reorient at rate k_orient; new informed pullers orient at once.
  # Between reorientations phi stays fixed relative to the site normal.
  pulling = (new_status == INFORMED) | (new_status == PULLER)
  reorient = (pulling & (u[3] < prob(p.k_orient))) | attach
  target = informed_targets(frame.pos, spec)
  desired = jnp.where((new_status == INFORMED)[:, None], target, f_loc)
  has_dir = jnp.sum(desired**2, -1) > 1e-12
  phi_new = clipped_angle(frame.normal, desired, phi_max)
  phi = jnp.where(reorient & has_dir, phi_new, jnp.where(attach, 0.0, phi))
  return new_status, phi


def population_update(key, status, near_wall, dt, p: AntParams):
  """Attachment, forgetting and detachment only, for policy-controlled ants.

  Roles and pull directions come from the policy, so role switching and
  reorientation are left out. Attached uninformed ants keep status PULLER
  whatever their role; the policy decides whether they pull.
  """
  u = jax.random.uniform(key, (2,) + status.shape)
  prob = lambda rate: 1.0 - jnp.exp(-rate * dt)
  status = jnp.where(near_wall, EMPTY, status)
  uninformed = (status == PULLER) | (status == LIFTER)
  forget = (status == INFORMED) & (u[0] < prob(p.k_forget))
  detach = uninformed & (u[0] < prob(p.k_off))
  attach = (status == EMPTY) & ~near_wall & (u[1] < prob(p.k_on))
  status = jnp.where(forget, PULLER, status)
  status = jnp.where(detach, EMPTY, status)
  return jnp.where(attach, INFORMED, status)


def is_pulling(status):
  return (status == INFORMED) | (status == PULLER)


def generalized_force(pulling, phi, frame: SiteFrame, p: AntParams):
  """Sum of puller forces as [F_x, F_y, torque about the COM].

  Args:
    pulling: (N_max,) bool, which sites currently pull.
  """
  f = p.f0 * pull_dirs(frame.normal, phi) * pulling[:, None]
  torque = jnp.sum(frame.r[:, 0] * f[:, 1] - frame.r[:, 1] * f[:, 0])
  return jnp.concatenate([f.sum(0), torque[None]])


def order_parameter(status, phi, frame: SiteFrame):
  """Generalized order parameter M (Materials and Methods, Eq. 1)."""
  pulling = (status == INFORMED) | (status == PULLER)
  v_hat = frame.vel / (jnp.linalg.norm(frame.vel, axis=-1, keepdims=True) +
                       1e-9)
  align = jnp.sum(pull_dirs(frame.normal, phi) * v_hat, -1)
  n_p = jnp.sum(pulling)
  return jnp.where(n_p > 0, jnp.sum(align * pulling) / jnp.maximum(n_p, 1), 0.)
