"""Discrete Soft Actor-Critic for the routing decision, Section IV-B.

Phase 4. Maximum-entropy RL with a stochastic policy, which the paper picks over DQN and
DDPG because the routing policy is not unique and a deterministic one generalises poorly
across deployments.

Table II: critic (256, 1024, 1024, 256), actor (128, 512, 512, 128), batch 1024,
learning rate 1e-3, gamma 0.99.
"""
from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from drl_routing.routing.environment import RoutingEnvironment

LOGIT_LIMIT = 8.0        # past this the softmax is saturated and the actor loses gradient
ALPHA_CEILING = 0.5      # keeps some exploration pressure without letting entropy dominate


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Actor(nn.Module):
    """The policy over neighbour slots."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state):
        # Squashed so no slot can saturate the softmax. Once a probability reaches one
        # the actor loss has almost no gradient left and the policy stops responding to
        # the critic, however far apart their values are.
        return LOGIT_LIMIT * torch.tanh(self.net(state) / LOGIT_LIMIT)

    def act(self, state, deterministic: bool = False, mask=None):
        logits = self.forward(state)
        if mask is not None:
            # A large negative rather than -inf: softmax over -inf gives exactly zero and
            # the log of that is a NaN gradient.
            logits = logits.masked_fill(~mask, -1e8)
        probs = F.softmax(logits, dim=-1)
        action = (torch.argmax(probs, dim=-1) if deterministic
                  else torch.multinomial(probs, 1).squeeze(-1))
        return action, probs, torch.log(probs.clamp(min=1e-8))


class Critic(nn.Module):
    """Q(s, .) over the neighbour slots.

    Larger than the actor, as Table II has it: predicting a value for every action is the
    harder of the two problems.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.data: deque = deque(maxlen=capacity)

    def push(self, *transition):
        self.data.append(transition)

    def sample(self, batch_size: int):
        s, a, r, s2, d, m, m2 = zip(*random.sample(self.data, batch_size))
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32),
                torch.tensor(np.array(m), dtype=torch.bool),
                torch.tensor(np.array(m2), dtype=torch.bool))

    def __len__(self):
        return len(self.data)


def train(env: RoutingEnvironment, ue_ids: list[int], episodes: int = 6000,
          gamma: float = 0.99, lr: float = 1e-3, batch_size: int = 1024,
          tau: float = 0.005, warm_start_fraction: float = 0.7,
          seed: int = 42, report_every: int = 500, verbose: bool = True):
    """Train the policy, warm-starting on the route the donor configured.

    Section IV has each node begin on the configured routing table and re-select
    neighbours as it learns. Ours follows that route with a probability that decays over
    the first warm_start_fraction of training, so the buffer fills with episodes that
    reach the donor before the policy is asked to find one itself.

    Returns the actor, a critic, and the mean return per reporting window.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = pick_device()
    if verbose:
        print(f"training on {device}")

    actor = Actor(env.state_dim, env.action_dim).to(device)
    q1 = Critic(env.state_dim, env.action_dim).to(device)
    q2 = Critic(env.state_dim, env.action_dim).to(device)
    q1_target = Critic(env.state_dim, env.action_dim).to(device)
    q2_target = Critic(env.state_dim, env.action_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)

    # The entropy target counts the neighbours a node actually has, not the padded action
    # space. Aiming at log(MAX_DEGREE) asks the policy to spread over slots that do not
    # exist, and alpha grows without bound trying to make it.
    mean_degree = np.mean([len(n) for n in env.neighbours.values()])
    target_entropy = 0.6 * math.log(max(mean_degree, 2.0))
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)

    buffer = ReplayBuffer()
    returns: list[float] = []
    window: list[float] = []

    for episode in range(episodes):
        state = env.reset(random.choice(ue_ids))
        episode_return = 0.0
        done = False

        while not done:
            mask_np = env.action_mask()
            mask = torch.tensor(mask_np).unsqueeze(0).to(device)

            follow = random.random() < max(
                0.0, 1.0 - episode / (warm_start_fraction * episodes))
            default = env.default_action(env.node) if follow else None
            if default is not None and mask_np[default]:
                action = default
            else:
                with torch.no_grad():
                    sampled, _, _ = actor.act(
                        torch.tensor(state).unsqueeze(0).to(device), mask=mask)
                action = int(sampled.item())

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, float(done),
                        mask_np, env.action_mask())
            state = next_state
            episode_return += reward

            if len(buffer) >= batch_size:
                S, A, R, S2, D, M, M2 = (t.to(device) for t in buffer.sample(batch_size))
                alpha = log_alpha.exp().detach()

                with torch.no_grad():
                    _, next_probs, next_logp = actor.act(S2, mask=M2)
                    # a masked slot carries no probability, so its log term must not
                    # enter the sum: log(1e-8) is -18 and inflates every target
                    next_logp = next_logp * (next_probs > 0)
                    min_q_next = torch.min(q1_target(S2), q2_target(S2))
                    value_next = (next_probs * (min_q_next - alpha * next_logp)).sum(-1)
                    target_q = R + gamma * (1 - D) * value_next

                q1_pred = q1(S).gather(1, A.unsqueeze(-1)).squeeze(-1)
                q2_pred = q2(S).gather(1, A.unsqueeze(-1)).squeeze(-1)
                critic_loss = (F.mse_loss(q1_pred, target_q)
                               + F.mse_loss(q2_pred, target_q))
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                _, probs, logp = actor.act(S, mask=M)
                min_q = torch.min(q1(S), q2(S))
                actor_loss = (probs * (alpha * logp - min_q)).sum(-1).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                entropy = -(probs * logp).sum(-1).mean()
                alpha_loss = -(log_alpha * (target_entropy - entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                with torch.no_grad():
                    log_alpha.clamp_(max=math.log(ALPHA_CEILING))

                for target, online in ((q1_target, q1), (q2_target, q2)):
                    for tp, op in zip(target.parameters(), online.parameters()):
                        tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

        window.append(episode_return)
        if (episode + 1) % report_every == 0:
            mean_return = float(np.mean(window))
            returns.append(mean_return)
            window = []
            if verbose:
                print(f"episode {episode + 1}/{episodes}  mean return {mean_return:6.2f}")

    return actor, q1, returns


def evaluate(env: RoutingEnvironment, actor: Actor, ue_ids: list[int]) -> list[dict]:
    """Run the trained policy from every UE and report the route it takes."""
    device = next(actor.parameters()).device
    rows = []
    for ue in ue_ids:
        state = env.reset(ue)
        path = [ue]
        done = False
        while not done:
            mask = torch.tensor(env.action_mask()).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = actor.act(torch.tensor(state).unsqueeze(0).to(device),
                                         deterministic=True, mask=mask)
            state, _, done, info = env.step(int(action.item()))
            path.append(env.node)
        rows.append({
            "ue": ue,
            "algorithm": "sac",
            "path": path,
            "hops": len(path) - 1,
            "reached": info["reached"],
            "delay_ms": info["delay_ms"] if info["reached"] else float("nan"),
            "reliability": info["reliability"] if info["reached"] else 0.0,
        })
    return rows
