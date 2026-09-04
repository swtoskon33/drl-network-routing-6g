"""AARL: PPO for real-time mmWave link scheduling.

Wraps the scheduling environment as a Gymnasium environment and trains a PPO agent on it,
following Gahtan, Cohen, Bronstein and Kedar, NoF 2023. The agent assigns a power to
every link in the mesh each slot; the point of the paper is that inference is a single
forward pass, so the decision fits inside the 10 ms slot regardless of topology size,
which the combinatorial baseline cannot manage.

Two agents, as in the paper: drop-insensitive at alpha = 1 and drop-sensitive at
alpha = 10.

    python scripts/train_scheduler.py --size small --steps 200000
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:      # the environment itself does not need gymnasium
    gym = None
    spaces = None
    GYM_AVAILABLE = False

from drl_routing.scheduling.env import SchedulingEnv
from drl_routing.scheduling.mesh import MmWaveMesh, SchedulingConfig, build_mesh


class SchedulingGymEnv(gym.Env if GYM_AVAILABLE else object):
    """Gymnasium wrapper so Stable-Baselines3 can train on the mesh.

    The action is a power in [0, 1] per link. The paper uses 0.01 steps for AARL against
    0.1 for the baseline, on the grounds that a neural network's runtime does not depend
    on the number of power levels while a search over them does; a continuous action
    space is the same argument taken to its limit.
    """

    metadata = {"render_modes": []}

    def __init__(self, size: str = "small", alpha: float = 10.0,
                 interference_level: float = 0.6, workload: str = "uniform",
                 initial_packets: int | None = None, seed: int = 42):
        super().__init__()
        packets = initial_packets or {"small": 2304, "medium": 10812, "large": 45246}[size]
        self.config = SchedulingConfig(interference_level=interference_level,
                                       initial_packets=packets, seed=seed)
        self.size = size
        self.alpha = alpha
        self.workload = workload
        self._build()

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(self.env.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.env.action_dim,), dtype=np.float32)

    def _build(self) -> None:
        mesh = build_mesh(self.size, self.config)
        self.env = SchedulingEnv(mesh, alpha=self.alpha, workload=self.workload)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state = self.env.reset()
        return state.astype(np.float32), {}

    def step(self, action):
        state, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        truncated = done and self.env.packets_in_system > 0
        return state.astype(np.float32), float(reward), bool(done and not truncated), \
            bool(truncated), info

    def summary(self) -> dict:
        return self.env.summary()


def train(size: str = "small", alpha: float = 10.0, interference_level: float = 0.6,
          steps: int = 200_000, seed: int = 42):
    """Train a PPO agent, with the network width the paper scales to topology size."""
    from stable_baselines3 import PPO

    env = SchedulingGymEnv(size=size, alpha=alpha, interference_level=interference_level,
                           seed=seed)
    # "MLP fully connected with different number of neurons per layer, according to the
    # topology's size", Fig. 5(b): 256 for the small mesh up to 4096 for the large one
    width = {"small": 256, "medium": 1024, "large": 4096}[size]
    model = PPO("MlpPolicy", env, learning_rate=3e-4, seed=seed, verbose=0,
                policy_kwargs={"net_arch": [width, width]})
    model.learn(total_timesteps=steps, progress_bar=False)
    return model, env


def evaluate(model, env: SchedulingGymEnv, episodes: int = 5) -> dict:
    """Run the trained policy and report goodput, slots used and decision time."""
    import time

    goodputs, slots, decision_times = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            started = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            decision_times.append((time.perf_counter() - started) * 1000)
            obs, _, done, truncated, _ = env.step(action)
        s = env.summary()
        goodputs.append(s["goodput"])
        slots.append(s["slots"])
    return {
        "goodput": float(np.mean(goodputs)),
        "slots": float(np.mean(slots)),
        "decision_ms": float(np.mean(decision_times)),
    }
