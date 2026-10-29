"""MJX simulator of the ant T-load puzzle (Dreyer et al., PNAS 2024)."""

import contextlib
import io

# MJX prints "Failed to import warp" on import when its optional NVIDIA Warp
# backend is absent. We use the JAX backend, so the notice is just noise.
with contextlib.redirect_stdout(io.StringIO()):
  from mujoco import mjx  # noqa: F401

from tmovers.env import EnvState, TMoversEnv, rollout  # noqa: E402
