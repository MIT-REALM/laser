import os
import jax
import cyclopts
import numpy as np
import yaml
import dataclasses
import jax.random as jr

from laser.trainer.utils import restore_agent, test_actor, supply_rng
from laser.agents import agents, agent_cfgs
from laser.env import make_env


app = cyclopts.App()


@app.default
def test(
        path: str,
        step: int | None = None,
        epi: int = 10,
        seed: int = 0,
        no_video: bool = False,
        debug: bool = False
):
    """
    Test a trained agent in the specified environment and generate videos.

    Parameters
    ----------
    path : str
        Path to the directory containing the trained model and config.yaml.
    step : int | None
        Specific training step to load the model from. If None, loads the latest model.
    epi : int
        Positive number of episodes to test.
    seed : int
        Random seed for testing.
    no_video : bool
        If True, do not generate videos.
    debug : bool
        Disable JIT compilation.
    """
    if isinstance(epi, bool) or not isinstance(epi, (int, np.integer)) or epi < 1:
        raise ValueError(f"epi must be a positive integer (got {epi!r}).")

    # Set up environment variables and seed.
    if debug:
        jax.config.update("jax_disable_jit", True)
    global_rng = np.random.default_rng(seed)
    np.random.seed(global_rng.integers(2**31))

    # Load config.
    with open(os.path.join(path, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    # Create environment.
    env = make_env(
        config['env_name'], frame_stack=config.get('frame_stack'),
        render_mode=None if no_video else "rgb_array", width=1080, height=720
    )

    try:
        # Load model.
        model_path = os.path.join(path, 'models')
        if step is not None:
            load_step = step
        else:
            models = os.listdir(model_path)
            load_step = max([int(m.split('_')[-1].split('.')[0]) for m in models if m.startswith('params_')])

        agent_name = config['agent_name']
        agent_class = agents[agent_name]
        agent_cfg_class = agent_cfgs[agent_name]
        valid_keys = {field.name for field in dataclasses.fields(agent_cfg_class)}
        agent_cfg = agent_cfg_class(**{key: value for key, value in config.items() if key in valid_keys})
        # Only input shapes are needed before restoring the trained parameters.
        example_observations = np.zeros((1, *env.observation_space.shape), dtype=env.observation_space.dtype)
        example_actions = np.zeros((1, *env.action_space.shape), dtype=env.action_space.dtype)
        agent = agent_class.create(
            global_rng.integers(2**31), example_observations, example_actions, agent_cfg,
        )
        agent = restore_agent(agent, model_path, load_step)
        print('Loaded model from:', model_path)

        # Test the agent in the environment.
        key = jr.PRNGKey(global_rng.integers(2**31))
        actor_fn = supply_rng(agent.sample_actions, key=key)
        videos_dir = None if no_video else os.path.join(path, 'videos', f'step_{load_step}')
        trajs, _ = test_actor(
            actor_fn,
            env,
            n_episodes=epi,
            rng=global_rng,
            render=not no_video,
            verbose=True,
            action_dim=env.action_space.shape[0],
            video_dir=videos_dir,
        )
    finally:
        env.close()

    reward_mean = np.mean([np.sum(traj['reward']) for traj in trajs])
    success_mean = np.mean([traj['info'][-1]['success'] for traj in trajs])
    print(f'Test results over {epi} episodes -- '
          f'Mean Reward: {reward_mean:.2f}, Mean Success: {success_mean:.2f}')

    if not no_video:
        print('Saved video to: ', videos_dir)


if __name__ == "__main__":
    app()
