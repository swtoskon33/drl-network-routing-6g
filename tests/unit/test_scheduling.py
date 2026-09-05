"""Tests for the scheduling mesh, its environment and the RPMA baseline."""
import numpy as np
import pytest

from drl_routing.scheduling.env import SchedulingEnv
from drl_routing.scheduling.mesh import SchedulingConfig, build_mesh
from drl_routing.scheduling.rpma import rpma_schedule


@pytest.mark.unit
@pytest.mark.parametrize(("size", "links"), [("small", 10), ("medium", 48), ("large", 96)])
def test_meshes_have_the_size_the_paper_evaluates(size, links):
    assert len(build_mesh(size).links) == links


@pytest.mark.unit
def test_interference_costs_capacity():
    """Two active links must be worth less than the sum of each alone, or there is no
    scheduling problem to solve."""
    mesh = build_mesh("small", SchedulingConfig(interference_level=0.8))
    n = len(mesh.links)
    first = np.zeros(n, dtype=np.float32); first[0] = 1.0
    second = np.zeros(n, dtype=np.float32); second[1] = 1.0
    both = first + second
    assert mesh.effective_capacity(both).sum() < (
        mesh.effective_capacity(first).sum() + mesh.effective_capacity(second).sum())


@pytest.mark.unit
def test_full_interference_makes_activating_everything_useless():
    mesh = build_mesh("small", SchedulingConfig(interference_level=1.0))
    everything = np.ones(len(mesh.links), dtype=np.float32)
    assert mesh.effective_capacity(everything).sum() == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_packets_move_and_the_episode_ends():
    env = SchedulingEnv(build_mesh("small", SchedulingConfig(initial_packets=500)))
    start = env.packets_in_system
    assert start > 0
    done = False
    while not done:
        _, _, done, _ = env.step(np.ones(env.action_dim, dtype=np.float32))
    assert env.delivered > 0
    assert env.summary()["goodput"] <= 1.0


@pytest.mark.unit
def test_reward_penalises_drops():
    """alpha weights drops against movement, so a drop-sensitive agent scores a slot with
    drops lower than a drop-insensitive one scores the same slot."""
    mesh = build_mesh("small", SchedulingConfig(initial_packets=500))
    sensitive = SchedulingEnv(mesh, alpha=10.0)
    insensitive = SchedulingEnv(mesh, alpha=1.0)
    action = np.ones(sensitive.action_dim, dtype=np.float32)
    _, r_sensitive, _, _ = sensitive.step(action)
    _, r_insensitive, _, _ = insensitive.step(action)
    assert r_sensitive <= r_insensitive


@pytest.mark.unit
def test_rpma_only_schedules_links_with_traffic():
    mesh = build_mesh("small")
    demands = np.zeros(len(mesh.links), dtype=np.float32)
    demands[0] = 100
    powers = rpma_schedule(mesh, demands)
    assert powers[0] > 0
    assert powers[1:].sum() == 0
