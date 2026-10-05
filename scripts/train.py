import os
import jax
import cyclopts
import numpy as np
import glob

from laser.trainer.utils import is_connected
from laser.agents import agents, AgentCfg, FQLCfg, IFQLCfg, DSRLCfg, ReFORMCfg, LASERCfg, QAMCfg, QAMECfg
from laser.trainer.datasets import Dataset, DatasetCfg
from laser.trainer.trainer import TrainerCfg, Trainer
from laser.env import make_env_and_datasets


app = cyclopts.App()


def _train(
        agent_cfg: AgentCfg,
        trainer_cfg: TrainerCfg,
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-single-noisy-singletask-task1-v0",
        seed: int = 0,
        debug: bool = False,
        dataset_path: str | None = None
):
    # Set up environment variables and seed.
    if debug:
        os.environ["WANDB_MODE"] = "disabled"
        jax.config.update("jax_disable_jit", True)
    elif not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    global_rng = np.random.default_rng(seed)
    np.random.seed(global_rng.integers(2 ** 31))

    # Create environment and load datasets.
    if dataset_path is not None:
        dataset_idx = 0
        dataset_paths = [file for file in sorted(glob.glob(f"{dataset_path}/*.npz")) if '-val.npz' not in file]
        if not dataset_paths:
            raise ValueError(f"No training datasets found in {dataset_path!r}.")
        env, train_dataset, val_dataset = make_env_and_datasets(
            env_name, frame_stack=dataset_cfg.frame_stack, render_mode="rgb_array",
            dataset_path=dataset_paths[dataset_idx]
        )
    else:
        dataset_paths = None
        dataset_idx = 0
        env, train_dataset, val_dataset = make_env_and_datasets(
            env_name, frame_stack=dataset_cfg.frame_stack, render_mode="rgb_array"
        )
    train_dataset = Dataset.create(dataset_cfg, global_rng.integers(2**31), **train_dataset)
    val_dataset = Dataset.create(dataset_cfg, global_rng.integers(2**31), **val_dataset)

    # Create agent.
    example_batch = train_dataset.sample(1)
    agent_class = agents[agent_cfg.agent_name]
    agent = agent_class.create(
        global_rng.integers(2**31),
        example_batch['observations'],
        example_batch['actions'],
        agent_cfg,
    )

    # Initialize trainer.
    trainer = Trainer(
        trainer_cfg, env_name, env, train_dataset, val_dataset, dataset_cfg, agent,
        dataset_paths=dataset_paths, dataset_idx=dataset_idx, rng=global_rng,
    )

    # Start training.
    try:
        trainer.train(seed, debug)
    finally:
        env.close()


@app.command
def fql(
        agent_cfg: FQLCfg = FQLCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-double-play-singletask-task2-v0",
        seed: int = 0,
        debug: bool = False,
        dataset_path: str | None = None
):
    """
    Train a FQL agent.

    Parameters
    ----------
    agent_cfg : FQLCfg
        Configuration for the FQL agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        Disable JIT compilation, logging, and checkpoint saving.
    dataset_path : str | None
        Directory of training .npz files and their corresponding -val.npz files; None uses OGBench data.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug, dataset_path)


@app.command
def ifql(
        agent_cfg: IFQLCfg = IFQLCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-double-play-singletask-task2-v0",
        seed: int = 0,
        debug: bool = False,
        dataset_path: str | None = None
):
    """
    Train a IFQL agent.

    Parameters
    ----------
    agent_cfg : IFQLCfg
        Configuration for the IFQL agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        Disable JIT compilation, logging, and checkpoint saving.
    dataset_path : str | None
        Directory of training .npz files and their corresponding -val.npz files; None uses OGBench data.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug, dataset_path)


@app.command
def dsrl(
        agent_cfg: DSRLCfg = DSRLCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-double-play-singletask-task2-v0",
        seed: int = 0,
        debug: bool = False,
        dataset_path: str | None = None
):
    """
    Train a DSRL agent.

    Parameters
    ----------
    agent_cfg : DSRLCfg
        Configuration for the DSRL agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        Disable JIT compilation, logging, and checkpoint saving.
    dataset_path : str | None
        Directory of training .npz files and their corresponding -val.npz files; None uses OGBench data.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug, dataset_path)


@app.command
def reform(
        agent_cfg: ReFORMCfg = ReFORMCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-double-play-singletask-task2-v0",
        seed: int = 0,
        debug: bool = False,
        dataset_path: str | None = None
):
    """
    Train a ReFORM agent.

    Parameters
    ----------
    agent_cfg : ReFORMCfg
        Configuration for the ReFORM agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        Disable JIT compilation, logging, and checkpoint saving.
    dataset_path : str | None
        Directory of training .npz files and their corresponding -val.npz files; None uses OGBench data.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug, dataset_path)


@app.command
def laser(
        agent_cfg: LASERCfg = LASERCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-double-play-singletask-task2-v0",
        seed: int = 0,
        debug: bool = False,
        dataset_path: str | None = None
):
    """
    Train a LASER agent.

    Parameters
    ----------
    agent_cfg : LASERCfg
        Configuration for the LASER agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        Disable JIT compilation, logging, and checkpoint saving.
    dataset_path : str | None
        Directory of training .npz files and their corresponding -val.npz files; None uses OGBench data.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug, dataset_path)


@app.command
def qam(
        agent_cfg: QAMCfg = QAMCfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-double-play-singletask-task2-v0",
        seed: int = 0,
        debug: bool = False,
        dataset_path: str | None = None
):
    """
    Train a QAM agent.

    Parameters
    ----------
    agent_cfg : QAMCfg
        Configuration for the QAM agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        Disable JIT compilation, logging, and checkpoint saving.
    dataset_path : str | None
        Directory of training .npz files and their corresponding -val.npz files; None uses OGBench data.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug, dataset_path)


@app.command
def qam_e(
        agent_cfg: QAMECfg = QAMECfg(),
        trainer_cfg: TrainerCfg = TrainerCfg(),
        dataset_cfg: DatasetCfg = DatasetCfg(),
        env_name: str = "cube-double-play-singletask-task2-v0",
        seed: int = 0,
        debug: bool = False,
        dataset_path: str | None = None
):
    """
    Train a QAME agent.

    Parameters
    ----------
    agent_cfg : QAMECfg
        Configuration for the QAME agent.
    trainer_cfg : TrainerCfg
        Configuration for the trainer.
    dataset_cfg : DatasetCfg
        Configuration for the dataset.
    env_name : str
        Name of the environment to train on.
    seed : int
        Random seed.
    debug : bool
        Disable JIT compilation, logging, and checkpoint saving.
    dataset_path : str | None
        Directory of training .npz files and their corresponding -val.npz files; None uses OGBench data.
    """
    _train(agent_cfg, trainer_cfg, dataset_cfg, env_name, seed, debug, dataset_path)


if __name__ == "__main__":
    app()
