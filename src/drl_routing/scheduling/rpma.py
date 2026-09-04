"""RPMA: the combinatorial baseline the learned scheduler is measured against.

Residual Profit Maximizer, Section V of Gahtan et al. It considers links one at a time in
random order and, for each, tries every power level. Adding a link earns capacity on that
link and costs capacity on every link already chosen, through interference; the
difference is the residual profit. A link joins the schedule at the power where that
profit is highest, and is left out when no power makes it positive.

It produces good schedules. What it cannot do is produce them in time: the cost is
O(E^2 * power levels) per slot, so on a 96-link mesh a decision takes longer than the
slot it is deciding for. That gap is the point of the comparison.
"""
from __future__ import annotations

import random

import numpy as np

from drl_routing.scheduling.mesh import MmWaveMesh

POWER_LEVELS = np.arange(0.0, 1.01, 0.1)   # 11 levels, as the paper uses for RPMA


def rpma_schedule(mesh: MmWaveMesh, demands: np.ndarray,
                  rng: random.Random | None = None) -> np.ndarray:
    """Choose a power for every link in the mesh for the next slot.

    demands[i] is the number of packets waiting on link i, which caps how much capacity
    on that link is worth anything.
    """
    rng = rng or random.Random(0)
    n = len(mesh.links)
    powers = np.zeros(n, dtype=np.float32)
    nominal = np.array([mesh.capacity[link] for link in mesh.links], dtype=np.float32)

    order = list(range(n))
    rng.shuffle(order)

    for link in order:
        if demands[link] <= 0:
            continue
        best_profit, best_power = 0.0, 0.0
        for power in POWER_LEVELS[1:]:
            trial = powers.copy()
            trial[link] = power

            # capacity gained on this link, capped by what is actually queued
            gain = min(nominal[link] * max(power - mesh.interference[:, link] @ powers, 0.0),
                       demands[link])

            # capacity lost on links already scheduled, from this link's interference
            active = powers > 0
            if active.any():
                before = np.clip(powers[active]
                                 - mesh.interference[:, active].T @ powers, 0.0, None)
                after = np.clip(trial[active]
                                - mesh.interference[:, active].T @ trial, 0.0, None)
                loss = float(((before - after) * nominal[active]).sum())
            else:
                loss = 0.0

            profit = gain - loss
            if profit > best_profit:
                best_profit, best_power = profit, power

        powers[link] = best_power

    return powers


def run_rpma(env, rng: random.Random | None = None) -> tuple[dict, float]:
    """Play an episode with RPMA, returning its summary and mean decision time in ms."""
    import time

    rng = rng or random.Random(0)
    env.reset()
    decision_times = []
    done = False
    while not done:
        demands = np.array([len(env.buffers.get(link, ())) for link in env.mesh.links],
                           dtype=np.float32)
        started = time.perf_counter()
        powers = rpma_schedule(env.mesh, demands, rng)
        decision_times.append((time.perf_counter() - started) * 1000)
        _, _, done, _ = env.step(powers)
    return env.summary(), float(np.mean(decision_times))
