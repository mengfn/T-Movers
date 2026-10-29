"""Multi-agent "ant colony" environment, faithful to the paper's ant model.

Each of the N_max attachment sites is an agent slot. As in the paper (SI Note
1), ants attach, forget the target, detach and let go near walls at the
model's rates; force magnitude (f0) and attachment site are fixed. What an ant
*decides* is exactly what the paper's carriers decide, and the policy takes
those decisions every `decision_dt` seconds:

  action[i] = (pull, reorient, angle)
    pull      in {0, 1}: pull with f0 (1) or lift (0)
    reorient  in {0, 1}: adopt `angle` (1) or keep the current pull angle (0);
              between reorientations the angle stays fixed relative to the
              site normal, as in the paper
    angle     in [-1, 1]: new pull direction, as a fraction of phi_max = 52 deg
              from the site's outward normal

Observations are what a paper ant senses, in the frame of its site normal n
and tangent t (so one shared policy fits every ant):

  [pulling, phi / phi_max, f_loc . n, f_loc . t, p . f_loc, informed,
   target . n, target . t]

with forces in units of F_ind * f0 (the model's sensitivity scale) and the
target (next exit) direction known only to informed ants. Extra senses that
paper ants lack can be switched on with `extra_senses`.

`controlled` chooses who follows the policy:
  "uninformed": informed ants follow the paper's rule (pull towards the exit,
                reorient at k_orient); the policy is the uninformed ants' rule.
  "all":        every attached ant follows the policy.
`mask` marks the ants the policy controls (and the loss should use).

The team reward is progress along the shortest collision-free path to the
goal (configuration-space distance, in d_cor) plus a solve bonus.

For credit assignment during training, `agent_reward` splits the expected
progress of each decision between the ants: the load is overdamped, so its
velocity is D^-1 F with F the sum of the ants' generalized forces F_i, and
ant i's share of the progress is -grad(d) . D^-1 F_i * decision_dt (in d_cor).
It uses privileged information (the distance field), so it is a training
signal only; the ants' observations stay as above.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from tmovers import ants, configspace
from tmovers.env import EnvState, TMoversEnv

ACTION_DIM = 3


@dataclass(frozen=True)
class ColonyConfig:
  arena: str = "large"
  n_sites: Optional[int] = None  # N_max; None keeps the arena's (120 / 15)
  dt: float = 0.01
  action_repeat: int = 10  # decision_dt = 0.1 s
  max_time: float = 1800.0
  controlled: str = "uninformed"  # or "all"
  # Any of "walls", "site", "pose": wall distances, position on the load, and
  # the load's pose and puzzle state. Paper ants have none of these.
  extra_senses: tuple = ()
  solve_bonus: float = 10.0
  time_penalty: float = 0.0  # per decision
  # Training-only start conditions (evaluation uses the paper's protocol:
  # initial pose, empty load). warm_start fills the load with the model's
  # equilibrium numbers of carriers (SI Eqs. 10-12); random_start is the
  # probability of starting from a random collision-free pose instead of the
  # initial one.
  warm_start: bool = False
  random_start: float = 0.0

  @property
  def decision_dt(self) -> float:
    return self.action_repeat * self.dt


class ColonyState(NamedTuple):
  env: EnvState
  pull: jax.Array  # (N,) bool, the ant at each site pulls
  dist: jax.Array  # configuration-space distance to the goal [cm]
  obs: jax.Array  # (N, obs_dim)
  mask: jax.Array  # (N,) bool, site holds an ant the policy controls
  global_state: jax.Array  # (G,) for a centralized critic
  reward: jax.Array
  agent_reward: jax.Array  # (N,) each ant's expected share of the progress
  done: jax.Array
  steps: jax.Array


class ColonyEnv:

  def __init__(self, config: ColonyConfig = ColonyConfig(),
               params: ants.AntParams = ants.AntParams()):
    assert config.controlled in ("uninformed", "all")
    self.config = config
    self.base = TMoversEnv(config.arena, config.dt, params, config.max_time,
                           config.n_sites)
    self.cspace = configspace.ConfigSpace(config.arena)
    self.spec = self.base.spec
    self.n_agents = self.base.n_sites
    self.params = params
    self.phi_max = jnp.deg2rad(params.phi_max_deg)
    self.f_scale = params.f_ind * params.f0
    self.v_ref = 0.05 * self.spec.load_length  # for the critic only
    pos = self.base.site_pos
    arc = jnp.arctan2(pos[:, 1], pos[:, 0])
    self.site_code = jnp.stack([jnp.sin(arc), jnp.cos(arc)], -1)
    self.d_start = float(self.cspace.distance(self.base.initial_pose))
    # Candidate random starts: free, unsolved poses connected to the goal.
    cs = self.cspace
    poses = cs.grid.poses().reshape(-1, 3)
    ok = (cs.free.ravel() & (cs.dist_np.ravel() > 0) &
          (cs.dist_np.ravel() < cs.dist_np.max()))
    self.start_poses = jnp.asarray(poses[ok][::7], jnp.float32)
    k_on, k_off, k_forget = params.k_on, params.k_off, params.k_forget
    n = self.n_agents
    n_att = n / (1 + k_off * k_forget / (k_on * (k_off + k_forget)))  # Eq. 12
    n_info = k_on * (n - n_att) / k_forget  # Eq. 11 at equilibrium
    self.p_attached, self.p_informed = n_att / n, n_info / n_att
    probe = self.reset(jax.random.PRNGKey(0))
    self.obs_dim = int(probe.obs.shape[-1])
    self.global_dim = int(probe.global_state.shape[-1])

  # API ----------------------------------------------------------------------

  def reset(self, key) -> ColonyState:
    cfg = self.config
    k_env, k_pose, k_pick, k_att, k_inf, k_pull = jax.random.split(key, 6)
    pose = self.base.initial_pose
    if cfg.random_start > 0:
      pick = self.start_poses[jax.random.randint(k_pick, (), 0,
                                                 len(self.start_poses))]
      use = jax.random.uniform(k_pose) < cfg.random_start
      pose = jnp.where(use, pick, pose)
    env_state = self.base.reset(k_env, pose)
    pull = jnp.zeros(self.n_agents, bool)
    if cfg.warm_start:
      frame = self.base.site_frame(env_state.data)
      free = self.base.wall_distance(frame.pos) >= self.params.wall_eps
      shape = (self.n_agents,)
      attached = free & jax.random.bernoulli(k_att, self.p_attached, shape)
      informed = attached & jax.random.bernoulli(k_inf, self.p_informed, shape)
      status = jnp.where(informed, ants.INFORMED,
                         jnp.where(attached, ants.PULLER, ants.EMPTY))
      target_phi = ants.clipped_angle(
          frame.normal, ants.informed_targets(frame.pos, self.spec),
          self.phi_max)
      pull = informed | (attached & jax.random.bernoulli(k_pull, 0.5, shape))
      env_state = env_state._replace(
          status=status.astype(env_state.status.dtype),
          phi=jnp.where(informed, target_phi, 0.0))
    dist = self.cspace.distance(env_state.data.qpos)
    return self._finish(env_state, pull, dist, jnp.zeros(()),
                        jnp.zeros(self.n_agents), jnp.zeros((), jnp.int32))

  def step(self, state: ColonyState, action: jax.Array) -> ColonyState:
    """Applies (N, 3) actions, then simulates `decision_dt` seconds."""
    base, p = self.base, self.params
    env_state = state.env
    controlled = self._controlled(env_state.status)
    pull = jnp.where(controlled, action[:, 0] > 0.5, state.pull)
    new_phi = jnp.clip(action[:, 2], -1.0, 1.0) * self.phi_max
    phi = jnp.where(controlled & (action[:, 1] > 0.5), new_phi, env_state.phi)
    scripted_informed = self.config.controlled == "uninformed"
    agent_reward = self._credit(env_state, pull, phi)

    def physics(_, carry):
      s, pull = carry
      frame = base.site_frame(s.data)
      near_wall = base.wall_distance(frame.pos) < p.wall_eps
      key, k_pop, k_orient, k_role = jax.random.split(s.key, 4)
      status = ants.population_update(k_pop, s.status, near_wall, base.dt, p)
      attached = status != ants.EMPTY
      new = attached & (s.status == ants.EMPTY)
      # Forgetting includes the paper's role draw (SI Eq. 14): the ant pulls
      # with probability sigmoid(2 p . f_loc / F_ind) until it next decides.
      forgot = (s.status == ants.INFORMED) & (status == ants.PULLER)
      _, d_lin = ants.damping(jnp.sum(attached), self.n_agents,
                              base.inertia_ratio, p)
      x = jnp.sum(ants.pull_dirs(frame.normal, s.phi) * d_lin * frame.vel,
                  -1) / self.f_scale
      draw = jax.random.uniform(k_role, status.shape) < jax.nn.sigmoid(2 * x)
      pull = jnp.where(forgot, draw, pull)
      # Newly attached ants are informed pullers aimed at the exit; informed
      # ants that follow the paper's rule re-aim at rate k_orient.
      target_phi = ants.clipped_angle(
          frame.normal, ants.informed_targets(frame.pos, self.spec),
          self.phi_max)
      re_aim = new
      if scripted_informed:
        fires = jax.random.uniform(k_orient, status.shape) < (
            1 - jnp.exp(-p.k_orient * base.dt))
        re_aim = re_aim | ((status == ants.INFORMED) & fires)
        pull = jnp.where(status == ants.INFORMED, True, pull)
      phi = jnp.where(re_aim, target_phi, s.phi)
      pull = (pull | new) & attached
      data = base.physics_step(s.data, frame, status, pull, phi)
      return base._advance(s, data, status, phi, key), pull

    env_state, pull = jax.lax.fori_loop(
        0, self.config.action_repeat, physics,
        (env_state._replace(phi=phi), pull))
    dist = self.cspace.distance(env_state.data.qpos)
    newly_solved = env_state.solved & ~state.env.solved
    reward = ((state.dist - dist) / self.cspace.d_cor +
              self.config.solve_bonus * newly_solved -
              self.config.time_penalty)
    return self._finish(env_state, pull, dist, reward, agent_reward,
                        state.steps + 1)

  def display_status(self, state: ColonyState) -> jax.Array:
    """Site status for the renderers: lifters are attached ants not pulling."""
    s = state.env.status
    return jnp.where((s != ants.EMPTY) & ~state.pull, ants.LIFTER, s)

  def metrics(self, state: ColonyState) -> dict:
    return {
        **self.base.metrics(state.env),
        "n_pullers": jnp.sum((state.env.status != ants.EMPTY) & state.pull),
        "distance": state.dist / self.cspace.d_cor,
        "puzzle_state": configspace.classify(self.spec, state.env.data.qpos),
    }

  # Internals ------------------------------------------------------------------

  def _controlled(self, status):
    if self.config.controlled == "all":
      return status != ants.EMPTY
    return (status == ants.PULLER) | (status == ants.LIFTER)

  def _finish(self, env_state, pull, dist, reward, agent_reward,
              steps) -> ColonyState:
    obs = self._observe(env_state, pull)
    mask = self._controlled(env_state.status)
    return ColonyState(env=env_state, pull=pull, dist=dist,
                       obs=obs * mask[:, None], mask=mask,
                       global_state=self._global_state(env_state, dist),
                       reward=reward, agent_reward=agent_reward,
                       done=env_state.done, steps=steps)

  def _credit(self, env_state: EnvState, pull, phi):
    """Each ant's expected progress over the next decision [d_cor]."""
    frame = self.base.site_frame(env_state.data)
    status = env_state.status
    attached = status != ants.EMPTY
    f = self.params.f0 * ants.pull_dirs(frame.normal, phi) * (
        pull & attached)[:, None]
    torque = frame.r[:, 0] * f[:, 1] - frame.r[:, 1] * f[:, 0]
    damping, _ = ants.damping(jnp.sum(attached), self.n_agents,
                              self.base.inertia_ratio, self.params)
    velocity = jnp.concatenate([f, torque[:, None]], -1) / damping
    descent = -self.cspace.gradient(env_state.data.qpos)
    return (velocity @ descent) * self.config.decision_dt / self.cspace.d_cor

  def _observe(self, env_state: EnvState, pull):
    """Per-ant observation in the frame of its site normal (n, t)."""
    frame = self.base.site_frame(env_state.data)
    n = frame.normal
    t = jnp.stack([-n[:, 1], n[:, 0]], -1)
    local = lambda v: jnp.stack([jnp.sum(v * n, -1), jnp.sum(v * t, -1)], -1)

    status = env_state.status
    n_att = jnp.sum(status != ants.EMPTY)
    _, d_lin = ants.damping(n_att, self.n_agents, self.base.inertia_ratio,
                            self.params)
    f_loc = d_lin * frame.vel / self.f_scale  # Eq. (2), in F_ind f0
    p_dir = ants.pull_dirs(n, env_state.phi)
    informed = status == ants.INFORMED
    target = ants.informed_targets(frame.pos, self.spec)
    target = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-9)

    parts = [pull[:, None], (env_state.phi / self.phi_max)[:, None],
             local(f_loc), jnp.sum(p_dir * f_loc, -1)[:, None],
             informed[:, None], local(target) * informed[:, None]]
    senses = self.config.extra_senses
    if "walls" in senses:
      w = self.spec.load_width
      d_here = self.base.wall_distance(frame.pos)
      d_ahead = self.base.wall_distance(frame.pos + n * w)
      parts.append(jnp.clip(jnp.stack([d_here, d_ahead], -1) / w, 0.0, 3.0))
    if "site" in senses:
      parts.append(self.site_code)
    if "pose" in senses:
      g = self._pose_features(env_state)
      parts.append(jnp.broadcast_to(g, (self.n_agents, g.shape[-1])))
    return jnp.concatenate([jnp.asarray(x, jnp.float32) for x in parts], -1)

  def _pose_features(self, env_state: EnvState):
    q = env_state.data.qpos
    x_end = self.spec.slit2 + self.spec.load_length
    puzzle_state = configspace.classify(self.spec, q)
    return jnp.concatenate([
        jnp.array([q[0] / x_end, q[1] / self.spec.breadth, jnp.sin(q[2]),
                   jnp.cos(q[2])]),
        jax.nn.one_hot(puzzle_state, len(configspace.STATES)),
    ])

  def _global_state(self, env_state: EnvState, dist):
    qvel = env_state.data.qvel
    status = env_state.status
    return jnp.concatenate([
        self._pose_features(env_state),
        jnp.array([qvel[0] / self.v_ref, qvel[1] / self.v_ref,
                   qvel[2] * self.cspace.r_av / self.v_ref,
                   jnp.mean(status != ants.EMPTY),
                   jnp.mean(status == ants.INFORMED), dist / self.d_start,
                   env_state.time / self.config.max_time]),
    ]).astype(jnp.float32)


# Reference policies -----------------------------------------------------------
# All take (env, state, key) and return (N, 3) actions.


def paper_policy(env: ColonyEnv, state: ColonyState, key) -> jax.Array:
  """The paper's carrier rules (SI Eqs. 1-2), expressed as a policy.

  Uses only the observation. Over one decision interval Δ:
    role:   a puller keeps pulling with prob. exp(-k_c e^{-x} Δ), a lifter
            starts pulling with prob. 1 - exp(-k_c e^{x} Δ), x = p . f_loc.
    aim:    pullers reorient with prob. 1 - exp(-k_orient Δ), towards f_loc
            (uninformed) or the exit (informed), within +-phi_max.
  """
  p, dt = env.params, env.config.decision_dt
  obs = state.obs
  pulling, f_n, f_t, x = obs[:, 0] > 0.5, obs[:, 2], obs[:, 3], obs[:, 4]
  informed, tgt_n, tgt_t = obs[:, 5] > 0.5, obs[:, 6], obs[:, 7]
  k1, k2 = jax.random.split(key)
  u = jax.random.uniform(k1, x.shape)
  p_pull = jnp.where(pulling, jnp.exp(-p.k_c * jnp.exp(-x) * dt),
                     1 - jnp.exp(-p.k_c * jnp.exp(x) * dt))
  pull = informed | (u < p_pull)
  aim_n = jnp.where(informed, tgt_n, f_n)
  aim_t = jnp.where(informed, tgt_t, f_t)
  has_aim = aim_n**2 + aim_t**2 > 1e-12
  reorient = pull & has_aim & (jax.random.uniform(k2, x.shape) <
                               1 - jnp.exp(-p.k_orient * dt))
  angle = jnp.clip(jnp.arctan2(aim_t, aim_n) / env.phi_max, -1.0, 1.0)
  return jnp.stack([pull, reorient, angle], -1).astype(jnp.float32)


def greedy_policy(env: ColonyEnv, state: ColonyState, key) -> jax.Array:
  """Every controlled ant pulls straight towards the next exit, always."""
  del key
  frame = env.base.site_frame(state.env.data)
  target = ants.informed_targets(frame.pos, env.spec)
  phi = ants.clipped_angle(frame.normal, target, env.phi_max)
  ones = jnp.ones(env.n_agents)
  return jnp.stack([ones, ones, phi / env.phi_max], -1)


def random_policy(env: ColonyEnv, state: ColonyState, key) -> jax.Array:
  del state
  k1, k2, k3 = jax.random.split(key, 3)
  n = env.n_agents
  return jnp.stack([
      jax.random.bernoulli(k1, 0.5, (n,)).astype(jnp.float32),
      jax.random.bernoulli(k2, 0.5, (n,)).astype(jnp.float32),
      jax.random.uniform(k3, (n,), minval=-1.0, maxval=1.0)
  ], -1)
