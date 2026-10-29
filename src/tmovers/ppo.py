"""MAPPO for the ant colony: a shared actor for every ant, a centralized critic.

Actor: per-ant observation -> pull/lift (Bernoulli), reorient (Bernoulli) and
a new pull angle (Gaussian, clipped to [-1, 1] by the environment; it only
matters, and only enters the log-probability, when the ant reorients). All
ants share the parameters, which suits a colony whose membership changes all
the time.
Critic: global state -> value of the team reward.
The PPO objective is averaged over attached ants only (via the env's mask).
"""

from dataclasses import dataclass
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tmovers.colony import ColonyEnv, ColonyState


@dataclass(frozen=True)
class PPOConfig:
  num_envs: int = 1024
  rollout_len: int = 64  # decisions per env per update (6.4 s of puzzle time)
  updates: int = 2000
  epochs: int = 4
  minibatches: int = 8
  lr: float = 3e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip: float = 0.2
  entropy_coef: float = 0.003  # on the pull and reorient choices only
  # Weight of each ant's own expected progress (ColonyState.agent_reward)
  # added to the shared team advantage, for credit assignment.
  local_coef: float = 1.0
  # KL penalty towards a prior actor (the behaviour-cloned paper rule), so
  # RL improves on the paper's ants within a trust region around them.
  prior_kl_coef: float = 0.0
  value_coef: float = 0.5
  max_grad_norm: float = 0.5
  hidden: int = 128
  seed: int = 0


LOG_STD_MIN, LOG_STD_MAX = -3.0, -0.5


class Actor(nn.Module):
  hidden: int

  @nn.compact
  def __call__(self, obs):
    x = nn.tanh(nn.Dense(self.hidden)(obs))
    x = nn.tanh(nn.Dense(self.hidden)(x))
    head = nn.Dense(3, kernel_init=nn.initializers.orthogonal(0.01))(x)
    log_std = self.param("log_std", nn.initializers.constant(-1.0), (1,))
    # Bounded so exploration noise cannot grow without limit (the env clips
    # angles to [-1, 1], so a huge spread would cost nothing).
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    pull_logit, reorient_logit, angle_mean = head[..., 0], head[..., 1], head[
        ..., 2]
    return (pull_logit, reorient_logit, angle_mean,
            jnp.broadcast_to(log_std, angle_mean.shape))


class Critic(nn.Module):
  hidden: int

  @nn.compact
  def __call__(self, g):
    x = nn.tanh(nn.Dense(self.hidden)(g))
    x = nn.tanh(nn.Dense(self.hidden)(x))
    return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)[..., 0]


def _bernoulli_log_prob(logit, x):
  return x * jax.nn.log_sigmoid(logit) + (1 - x) * jax.nn.log_sigmoid(-logit)


def _bernoulli_entropy(logit):
  p = jax.nn.sigmoid(logit)
  return -(p * jax.nn.log_sigmoid(logit) + (1 - p) * jax.nn.log_sigmoid(-logit))


def log_prob(actor_out, raw_action):
  """Log-probability per ant of raw actions (pull, reorient, angle sample)."""
  pull_logit, reorient_logit, mean, log_std = actor_out
  pull, reorient, angle = (raw_action[..., i] for i in range(3))
  lp_angle = (-0.5 * ((angle - mean) / jnp.exp(log_std))**2 - log_std -
              0.5 * jnp.log(2 * jnp.pi))
  return (_bernoulli_log_prob(pull_logit, pull) +
          _bernoulli_log_prob(reorient_logit, reorient) + reorient * lp_angle)


def kl_divergence(out, prior_out):
  """KL(policy || prior) per ant; the angle term counts when reorienting."""
  def bern_kl(logit, prior_logit):
    p = jax.nn.sigmoid(logit)
    return (p * (jax.nn.log_sigmoid(logit) - jax.nn.log_sigmoid(prior_logit)) +
            (1 - p) * (jax.nn.log_sigmoid(-logit) -
                       jax.nn.log_sigmoid(-prior_logit)))
  pull, reorient, mean, log_std = out
  pull0, reorient0, mean0, log_std0 = prior_out
  angle_kl = (log_std0 - log_std + (jnp.exp(2 * log_std) + (mean - mean0)**2) /
              (2 * jnp.exp(2 * log_std0)) - 0.5)
  return (bern_kl(pull, pull0) + bern_kl(reorient, reorient0) +
          jax.nn.sigmoid(reorient) * angle_kl)


def entropy(actor_out):
  """Entropy of the discrete choices (the angle's spread is bounded instead)."""
  pull_logit, reorient_logit, _, _ = actor_out
  return _bernoulli_entropy(pull_logit) + _bernoulli_entropy(reorient_logit)


def sample(actor_out, key):
  pull_logit, reorient_logit, mean, log_std = actor_out
  k1, k2, k3 = jax.random.split(key, 3)
  pull = jax.random.bernoulli(k1, jax.nn.sigmoid(pull_logit))
  reorient = jax.random.bernoulli(k2, jax.nn.sigmoid(reorient_logit))
  angle = mean + jnp.exp(log_std) * jax.random.normal(k3, mean.shape)
  return jnp.stack([pull.astype(jnp.float32), reorient.astype(jnp.float32),
                    angle], -1)


def make_policy(actor_params, hidden: int = 128, deterministic: bool = True):
  """A `policy(state, key) -> actions` for evaluate.evaluate_policy."""
  actor = Actor(hidden)

  def policy(state: ColonyState, key):
    out = actor.apply(actor_params, state.obs)
    if deterministic:
      pull_logit, reorient_logit, mean, _ = out
      return jnp.stack([(pull_logit > 0).astype(jnp.float32),
                        (reorient_logit > 0).astype(jnp.float32), mean], -1)
    return sample(out, key)

  return policy


class Transition(NamedTuple):
  obs: jax.Array  # (B, N, D)
  mask: jax.Array  # (B, N)
  global_state: jax.Array  # (B, G)
  action: jax.Array  # (B, N, 3)
  log_prob: jax.Array  # (B, N)
  value: jax.Array  # (B,)
  reward: jax.Array  # (B,)
  agent_reward: jax.Array  # (B, N)
  done: jax.Array  # (B,) episode ended (solved or time limit)
  truncated: jax.Array  # (B,) ended by the time limit only
  final_value: jax.Array  # (B,) value of the state the episode ended in
  solved: jax.Array  # (B,) episode ended solved


class TrainState(NamedTuple):
  actor: dict
  critic: dict
  opt_state: optax.OptState
  env_state: ColonyState
  key: jax.Array


def make_train(env: ColonyEnv, cfg: PPOConfig, prior_params=None):
  """Returns (init, update): pure functions to jit / scan over."""
  actor, critic = Actor(cfg.hidden), Critic(cfg.hidden)
  tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                   optax.adam(cfg.lr, eps=1e-5))

  def init(key, actor_params=None) -> TrainState:
    k_actor, k_critic, k_env, key = jax.random.split(key, 4)
    if actor_params is None:
      actor_params = actor.init(k_actor, jnp.zeros((1, env.obs_dim)))
    critic_params = critic.init(k_critic, jnp.zeros((1, env.global_dim)))
    env_state = jax.vmap(env.reset)(jax.random.split(k_env, cfg.num_envs))
    opt_state = tx.init((actor_params, critic_params))
    return TrainState(actor_params, critic_params, opt_state, env_state, key)

  def env_step(carry, _):
    ts: TrainState = carry
    key, k_act, k_reset = jax.random.split(ts.key, 3)
    s = ts.env_state
    out = actor.apply(ts.actor, s.obs)
    action = sample(out, k_act)
    logp = log_prob(out, action)
    value = critic.apply(ts.critic, s.global_state)
    s2 = jax.vmap(env.step)(s, action)
    # Auto-reset finished puzzles.
    fresh = jax.vmap(env.reset)(jax.random.split(k_reset, cfg.num_envs))
    done = s2.done
    s_next = jax.tree.map(
        lambda a, b: jnp.where(done.reshape(done.shape + (1,) * (a.ndim - 1)),
                               b, a), s2, fresh)
    truncated = done & ~s2.env.solved
    final_value = critic.apply(ts.critic, s2.global_state)
    t = Transition(s.obs, s.mask, s.global_state, action, logp, value,
                   s2.reward, s2.agent_reward, done, truncated, final_value,
                   s2.env.solved & done)
    return ts._replace(env_state=s_next, key=key), t

  def gae(traj: Transition, last_value):

    def step(carry, t):
      adv, next_value = carry
      # A time limit is not a real ending: bootstrap from the final state.
      next_value = jnp.where(t.truncated, t.final_value, next_value)
      terminal = t.done & ~t.truncated
      delta = t.reward + cfg.gamma * next_value * (1 - terminal) - t.value
      adv = delta + cfg.gamma * cfg.gae_lambda * (1 - t.done) * adv
      return (adv, t.value), adv

    _, adv = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value), traj,
                          reverse=True)
    return adv, adv + traj.value

  def loss_fn(params, batch, adv, ret):
    actor_params, critic_params = params
    out = actor.apply(actor_params, batch.obs)
    logp = log_prob(out, batch.action)
    ratio = jnp.exp(logp - batch.log_prob)
    m = batch.mask.astype(jnp.float32)
    n = jnp.maximum(m.sum(), 1.0)
    team = ((adv - adv.mean()) / (adv.std() + 1e-8))[..., None]
    own = batch.agent_reward
    own_mean = jnp.sum(own * m) / n
    own_std = jnp.sqrt(jnp.sum((own - own_mean)**2 * m) / n) + 1e-8
    a = team + cfg.local_coef * (own - own_mean) / own_std
    pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * a)
    pg_loss = jnp.sum(pg * m) / n
    ent = jnp.sum(entropy(out) * m) / n
    v = critic.apply(critic_params, batch.global_state)
    v_loss = jnp.mean((v - ret)**2)
    total = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent
    info = {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent}
    if prior_params is not None:
      prior_out = jax.lax.stop_gradient(actor.apply(prior_params, batch.obs))
      kl = jnp.sum(kl_divergence(out, prior_out) * m) / n
      total = total + cfg.prior_kl_coef * kl
      info["prior_kl"] = kl
    return total, info

  def update(ts: TrainState, _):
    ts, traj = jax.lax.scan(env_step, ts, None, length=cfg.rollout_len)
    last_value = critic.apply(ts.critic, ts.env_state.global_state)
    adv, ret = gae(traj, last_value)

    def epoch(carry, _):
      params, opt_state, key = carry
      key, sub = jax.random.split(key)
      perm = jax.random.permutation(sub, cfg.num_envs).reshape(
          cfg.minibatches, -1)

      def minibatch(carry, idx):
        params, opt_state = carry
        take = lambda x: x[:, idx]
        batch = jax.tree.map(take, traj)
        grads, info = jax.grad(loss_fn, has_aux=True)(params, batch,
                                                      take(adv), take(ret))
        updates, opt_state = tx.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), info

      (params, opt_state), info = jax.lax.scan(minibatch, (params, opt_state),
                                               perm)
      return (params, opt_state, key), info

    key, sub = jax.random.split(ts.key)
    (params, opt_state, _), info = jax.lax.scan(
        epoch, ((ts.actor, ts.critic), ts.opt_state, sub), None,
        length=cfg.epochs)
    ts = ts._replace(actor=params[0], critic=params[1], opt_state=opt_state,
                     key=key)
    metrics = jax.tree.map(jnp.mean, info)
    metrics.update(
        reward=traj.reward.mean(),
        episodes=traj.done.sum(),
        solved=traj.solved.sum(),
        attached=traj.mask.sum(-1).mean(),
        pull_frac=jnp.sum(traj.action[..., 0] * traj.mask) /
        jnp.maximum(traj.mask.sum(), 1),
        distance=ts.env_state.dist.mean() / env.cspace.d_cor,
    )
    return ts, metrics

  return init, update


def behavior_cloning(env: ColonyEnv, expert, cfg: PPOConfig, iters: int = 200,
                     num_envs: int = 256, rollout_len: int = 32,
                     lr: float = 1e-3, seed: int = 0):
  """Fits the actor to an expert policy(state, key) by maximum likelihood.

  Used to start PPO from the paper's carrier rule (colony.paper_policy):
  rollouts follow the expert, and the actor learns its action distribution
  on the controlled ants.
  """
  actor = Actor(cfg.hidden)
  tx = optax.adam(lr)
  key = jax.random.PRNGKey(seed)
  key, k_init, k_env = jax.random.split(key, 3)
  params = actor.init(k_init, jnp.zeros((1, env.obs_dim)))
  opt_state = tx.init(params)
  state = jax.vmap(env.reset)(jax.random.split(k_env, num_envs))

  @jax.jit
  def collect(state, key):

    def step(carry, _):
      s, key = carry
      key, k_act, k_reset = jax.random.split(key, 3)
      action = jax.vmap(expert)(s, jax.random.split(k_act, num_envs))
      s2 = jax.vmap(env.step)(s, action)
      fresh = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
      s2 = jax.tree.map(
          lambda a, b: jnp.where(
              s2.done.reshape(s2.done.shape + (1,) * (a.ndim - 1)), b, a), s2,
          fresh)
      return (s2, key), (s.obs, s.mask, action)

    return jax.lax.scan(step, (state, key), None, length=rollout_len)

  def loss_fn(params, obs, mask, action):
    out = actor.apply(params, obs)
    lp = log_prob(out, action)
    m = mask.astype(jnp.float32)
    pull_logit = out[0]
    agree = jnp.sum(((pull_logit > 0) == (action[..., 0] > 0.5)) * m)
    return -jnp.sum(lp * m) / jnp.maximum(m.sum(), 1), agree / jnp.maximum(
        m.sum(), 1)

  @jax.jit
  def fit(params, opt_state, obs, mask, action):
    (loss, agree), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, obs, mask, action)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss, agree

  history = []
  for i in range(iters):
    key, sub = jax.random.split(key)
    (state, _), (obs, mask, action) = collect(state, sub)
    for _ in range(4):
      params, opt_state, loss, agree = fit(params, opt_state, obs, mask,
                                           action)
    history.append((float(loss), float(agree)))
  return params, history


def train(env: ColonyEnv, cfg: PPOConfig, log_every: int = 10, callback=None,
          time_budget: float | None = None, actor_params=None,
          prior_params=None):
  """Runs cfg.updates PPO updates, logging every `log_every` updates.

  Stops early once `time_budget` seconds of wall-clock time have passed.
  """
  import time
  start = time.perf_counter()
  init, update = make_train(env, cfg, prior_params)
  ts = jax.jit(init)(jax.random.PRNGKey(cfg.seed), actor_params)
  run = jax.jit(lambda ts: jax.lax.scan(update, ts, None, length=log_every))
  history = []
  for i in range(0, cfg.updates, log_every):
    ts, metrics = run(ts)
    metrics = jax.tree.map(lambda x: np.asarray(x), metrics)
    row = {k: float(v.sum() if k in ("episodes", "solved") else v.mean())
           for k, v in metrics.items()}
    row["update"] = i + log_every
    history.append(row)
    if callback is not None:
      callback(row, ts)
    if time_budget is not None and time.perf_counter() - start > time_budget:
      break
  return ts, history
