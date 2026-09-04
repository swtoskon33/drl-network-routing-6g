"""Ray Tune sweep over the scheduler's hyperparameters.

The paper fixes its hyperparameters; this searches them, which is what you would do
before fixing them. Each trial trains a PPO agent and reports the goodput it reaches, and
Tune runs the trials in parallel across the cores available.

    python scripts/tune_scheduler.py --samples 40
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def trial(config: dict) -> None:
    """One training run at the sampled hyperparameters."""
    from ray import train as ray_train
    from stable_baselines3 import PPO

    from drl_routing.scheduling.aarl import SchedulingGymEnv, evaluate

    env = SchedulingGymEnv(size=config["size"], alpha=config["alpha"],
                           interference_level=config["interference"])
    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                gamma=config["gamma"],
                ent_coef=config["ent_coef"],
                policy_kwargs={"net_arch": [config["width"], config["width"]]})
    model.learn(total_timesteps=config["steps"], progress_bar=False)
    result = evaluate(model, env, episodes=3)
    ray_train.report({"goodput": result["goodput"],
                      "slots": result["slots"],
                      "decision_ms": result["decision_ms"]})


def main() -> None:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--size", default="small")
    parser.add_argument("--steps", type=int, default=30_000)
    args = parser.parse_args()

    space = {
        "size": args.size,
        "steps": args.steps,
        "alpha": tune.choice([1.0, 5.0, 10.0]),
        "interference": tune.choice([0.4, 0.6, 0.8]),
        "learning_rate": tune.loguniform(1e-4, 3e-3),
        "n_steps": tune.choice([512, 1024, 2048]),
        "batch_size": tune.choice([64, 128, 256]),
        "gamma": tune.choice([0.95, 0.99]),
        "ent_coef": tune.loguniform(1e-4, 1e-1),
        "width": tune.choice([128, 256, 512]),
    }

    tuner = tune.Tuner(
        trial,
        param_space=space,
        tune_config=tune.TuneConfig(
            metric="goodput",
            mode="max",
            num_samples=args.samples,
            # stop the trials that are clearly behind rather than running every one to
            # the end; most of the budget goes to the configurations that look promising
            scheduler=ASHAScheduler(max_t=1, grace_period=1),
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="goodput", mode="max")
    print("best goodput:", round(best.metrics["goodput"], 3))
    print("best config:", {k: v for k, v in best.config.items() if k != "size"})


if __name__ == "__main__":
    main()
