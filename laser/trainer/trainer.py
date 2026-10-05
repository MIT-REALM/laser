import dataclasses
import yaml
import wandb
import os
import datetime
import tempfile
import gymnasium as gym
import numpy as np
import jax.random as jr

from dataclasses import dataclass
from cyclopts import Parameter
from tqdm import tqdm

from laser.env import make_env_and_datasets
from laser.trainer.datasets import Dataset, DatasetCfg
from laser.agents import Agent
from laser.trainer.utils import supply_rng, test_actor, save_agent, to_plain_data, get_dependency_versions
from laser.utils.typing import PRNGKey


@Parameter(name="*", group="TrainerConfig")
@dataclass
class TrainerCfg:
    """Configuration for the trainer.

    Parameters
    ----------
    steps : int
        Number of optimizer updates; must be a nonnegative integer.
    log_interval : int
        Positive interval (in completed training steps) for logging training metrics.
    eval_interval : int
        Positive interval (in completed training steps) for evaluation on the validation dataset.
    test_interval : int
        Positive interval (in completed training steps) for testing in the environment.
    test_epi : int
        Number of episodes for testing.
    save_interval : int
        Positive interval (in completed training steps) for saving the model.
    batch_size : int
        Batch size for training.
    save_dir : str
        Directory to save logs and models.
    run_name : str | None
        Optional name for the run.
    """
    steps: int = 2000000
    log_interval: int = 1000
    eval_interval: int = 10000
    test_interval: int = 50000
    test_epi: int = 20
    save_interval: int = 100000
    batch_size: int = 256
    save_dir: str = "logs"
    run_name: str | None = None

    def __post_init__(self):
        for name in (
                "steps", "log_interval", "eval_interval", "test_interval", "save_interval", "test_epi", "batch_size"):
            value = getattr(self, name)
            minimum = 0 if name == "steps" else 1
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                requirement = "nonnegative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {requirement} integer (got {value!r}).")


class Trainer:

    def __init__(
            self,
            cfg: TrainerCfg,
            env_name: str,
            env: gym.Env,
            train_dataset: Dataset,
            val_dataset: Dataset,
            dataset_cfg: DatasetCfg,
            agent: Agent,
            dataset_paths: list | None = None,
            dataset_idx: int = 0,
            dataset_replace_interval: int = 1000,
            rng: np.random.Generator | None = None
    ):
        self.cfg = cfg
        self.env_name = env_name
        self.env = env
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dataset_cfg = dataset_cfg
        self.agent = agent
        self.dataset_paths = dataset_paths
        self.dataset_idx = dataset_idx
        self.dataset_replace_interval = dataset_replace_interval
        self.rng = rng

    def train(self, seed: int, debug: bool = False):
        start_time = datetime.datetime.now()
        rng = self.rng if self.rng is not None else np.random.default_rng(seed)
        # Keep training randomness independent of evaluation frequency.
        key, eval_key, test_key = jr.split(jr.PRNGKey(seed), 3)

        run_name = self.cfg.run_name
        if run_name is None and hasattr(self.agent.cfg, "alpha"):
            run_name = f"alpha{self.agent.cfg.alpha}"
        run_dir = f"seed{seed}_{start_time.strftime('%m%d%H%M%S')}"
        if run_name:
            run_dir += f"_{run_name}"
        log_root = os.path.join(self.cfg.save_dir, self.env_name, self.agent.cfg.agent_name)

        if not debug:
            os.makedirs(log_root, exist_ok=True)
            # Atomically reserve a unique directory for each run.
            log_dir = tempfile.mkdtemp(prefix=f"{run_dir}_", dir=log_root)
            model_dir = os.path.join(log_dir, "models")
            os.makedirs(model_dir)
            config = dataclasses.asdict(self.cfg)
            config.update(dataclasses.asdict(self.agent.cfg))
            config.update(dataclasses.asdict(self.dataset_cfg))
            dataset_source = {"type": "ogbench", "name": self.env_name}
            if self.dataset_paths is not None:
                dataset_source = {
                    "type": "local",
                    "paths": [os.path.abspath(path) for path in self.dataset_paths],
                    "initial_index": self.dataset_idx,
                    "replace_interval": self.dataset_replace_interval,
                }
            config.update({
                "env_name": self.env_name,
                "agent_name": self.agent.cfg.agent_name,
                "seed": seed,
                "run_name": run_name,
                "dataset_source": dataset_source,
                "dependency_versions": get_dependency_versions(),
            })
            config = to_plain_data(config)
            with open(os.path.join(log_dir, "config.yaml"), "w") as file:
                yaml.safe_dump(config, file, sort_keys=False)
            wandb_name = f"seed{seed}_{self.agent.cfg.agent_name}"
            if run_name:
                wandb_name += f"_{run_name}"
            wandb.init(project="laser", dir=log_dir, group=self.env_name, name=wandb_name, config=config)

        steps_len = len(str(self.cfg.steps))
        with tqdm(total=self.cfg.steps, desc="Training", ncols=80) as pbar:
            for step in range(self.cfg.steps + 1):
                info = {}
                # Step zero evaluates the initial agent; later steps each complete one update.
                if step > 0:
                    pbar.set_description("Training")
                    batch = self.train_dataset.sample_sequence(
                        self.cfg.batch_size, self.agent.cfg.horizon_length, self.agent.cfg.gamma)
                    key, update_key = jr.split(key)
                    self.agent, update_info = self.agent.update(batch, update_key)
                    if step % self.cfg.log_interval == 0 or step == self.cfg.steps:
                        info.update({f"train/{k}": float(v) for k, v in update_info.items()})
                    pbar.update(1)

                # Test the agent in the environment.
                if step % self.cfg.test_interval == 0:
                    pbar.set_description("Testing")
                    test_key, cur_test_key = jr.split(test_key)
                    test_info = self.test(cur_test_key)
                    info.update(test_info)
                    metrics = ", ".join(f"{k}: {v:8.2f}" for k, v in test_info.items())
                    tqdm.write(f"Step: {step:>{steps_len}}, {metrics}")

                # Evaluate on the validation dataset.
                if step % self.cfg.eval_interval == 0:
                    pbar.set_description("Evaluating")
                    eval_key, cur_eval_key = jr.split(eval_key)
                    val_batch = self.val_dataset.sample_sequence(
                        self.cfg.batch_size, self.agent.cfg.horizon_length, self.agent.cfg.gamma)
                    _, val_info = self.agent.total_loss(batch=val_batch, params=None, key=cur_eval_key)
                    info.update({f"eval/{k}": float(v) for k, v in val_info.items()})

                # Checkpoints contain exactly `step` completed updates, including the final step.
                if not debug:
                    if step % self.cfg.save_interval == 0 or step == self.cfg.steps:
                        pbar.set_description("Saving model")
                        save_agent(self.agent, model_dir, step)
                    if info:
                        wandb.log(info, step=step)

                # Replace datasets only after a full interval, before the next training step.
                if (self.dataset_paths is not None and len(self.dataset_paths) > 1 and
                        self.dataset_replace_interval > 0 and 0 < step < self.cfg.steps and
                        step % self.dataset_replace_interval == 0):
                    self.dataset_idx = (self.dataset_idx + 1) % len(self.dataset_paths)
                    dataset_path = self.dataset_paths[self.dataset_idx]
                    train_dataset, val_dataset = make_env_and_datasets(
                        self.env_name, dataset_path=dataset_path, dataset_only=True, cur_env=self.env)
                    self.train_dataset = Dataset.create(self.dataset_cfg, rng.integers(2 ** 31), **train_dataset)
                    self.val_dataset = Dataset.create(self.dataset_cfg, rng.integers(2 ** 31), **val_dataset)
                    tqdm.write(f"Step: {step:>{steps_len}}, Replaced training dataset with {dataset_path}")

    def test(self, key: PRNGKey):
        """Test the agent in the environment."""
        actor_fn = supply_rng(self.agent.sample_actions, key=key)

        trajs, _ = test_actor(
            actor_fn,
            self.env,
            n_episodes=self.cfg.test_epi,
            rng=np.random.default_rng(int(sum(key))),
            render=False,
            verbose=False,
            action_dim=self.env.action_space.shape[0],
        )

        traj_rewards = [np.sum(traj['reward']) for traj in trajs]
        traj_success = [traj['info'][-1]['success'] for traj in trajs]

        return {
            'test/reward_mean': np.mean(traj_rewards),
            'test/reward_max': np.max(traj_rewards),
            'test/reward_min': np.min(traj_rewards),
            'test/success_rate': np.mean(traj_success),
        }
