import socket
import jax.numpy as jnp
import jax.tree_util as jtu
import gymnasium as gym
import numpy as np
import jax.random as jr
import flax
import os
import pickle
import platform
import datetime
import cv2

from importlib.metadata import PackageNotFoundError, version
from collections import defaultdict, deque
from typing import Callable
from tqdm import tqdm


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False


def is_connected():
    return internet()


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def has_inf(x):
    return jtu.tree_map(lambda y: jnp.isinf(y).any(), x)


def has_any_inf(x):
    return jnp.array(jtu.tree_flatten(has_inf(x))[0]).any()


def has_any_nan_or_inf(x):
    return has_any_nan(x) | has_any_inf(x)


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def clip_top_k_percent_grads(grad, k: float):
    """
    Clips the gradients with the top k% largest norms to match the norm
    at the exact k% threshold.
    """
    norms = jnp.linalg.norm(grad, axis=-1, keepdims=True)
    percentile_threshold = 100.0 - k
    max_allowed_norm = jnp.percentile(norms, percentile_threshold)

    safe_norms = jnp.maximum(norms, 1e-12)
    scaling_factors = jnp.where(
        norms > max_allowed_norm,
        max_allowed_norm / safe_norms,
        1.0
    )
    clipped_grads = grad * scaling_factors

    return clipped_grads


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def test_actor(
        actor_fn: Callable,
        env: gym.Env,
        n_episodes: int,
        rng: np.random.Generator,
        video_frame_skip: int = 3,
        render: bool = False,
        verbose: bool = False,
        action_dim: int = None,
        video_dir: str | None = None,
):
    """Return episode trajectories and paths of streamed videos.

    When render=True, video_dir is required and frames are encoded immediately.
    The caller owns the environment; this function closes its video writers.
    """
    trajs = []
    video_paths = []
    video_writer = None
    if action_dim is None:
        action_dim = env.action_space.shape[0]

    if render:
        if video_dir is None:
            raise ValueError("video_dir is required when render=True.")
        os.makedirs(video_dir, exist_ok=True)
        stamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        video_episode = None

        def frame_callback(frame, episode_index):
            nonlocal video_writer, video_episode
            if episode_index != video_episode:
                if video_writer is not None:
                    video_writer.release()
                episode_path = os.path.join(video_dir, f'epi{episode_index}_{stamp_str}.mp4')
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter(
                    episode_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height)
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {episode_path!r}.")
                video_episode = episode_index
                video_paths.append(episode_path)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    pbar = tqdm(total=n_episodes, desc="Testing", ncols=80) if verbose else None
    print_len = len(str(n_episodes))
    try:
        for i_traj in range(n_episodes):
            obs, info = env.reset(seed=int(rng.integers(2**31)))
            done = False
            traj = defaultdict(list)
            step = 0
            action_queue = deque()
            while not done:
                if not action_queue:
                    actions = np.asarray(actor_fn(obs)).reshape(-1, action_dim)
                    action_queue.extend(actions)
                action = action_queue.popleft()
                action = np.clip(action, -1, 1)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                transition = dict(
                    observation=obs,
                    next_observation=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    info=info,
                )
                add_to(traj, transition)

                if render and (step % video_frame_skip == 0 or done):
                    frame_callback(env.render(), i_traj)
                obs = next_obs

            trajs.append(traj)
            if verbose:
                tqdm.write(f"Episode {i_traj:>{print_len}}: "
                           f"Success: {str(bool(traj['info'][-1]['success'])).rjust(5)}, "
                           f"Reward: {jnp.sum(jnp.array(traj['reward'])):8.2f}, "
                           f"Length: {len(traj['reward']):>4}")
                pbar.update(1)
    finally:
        if video_writer is not None:
            video_writer.release()
        if pbar is not None:
            pbar.close()

    # Add episode metrics only after the video writers have closed.
    for i, episode_path in enumerate(video_paths):
        episode_name = (f'epi{i}_reward{int(np.sum(trajs[i]["reward"]))}_'
                        f'success{int(trajs[i]["info"][-1]["success"])}_{stamp_str}.mp4')
        video_paths[i] = os.path.join(video_dir, episode_name)
        os.rename(episode_path, video_paths[i])
    return trajs, video_paths


def supply_rng(f, key=None):
    """Helper function to split the random number generator key before each call to the function."""

    if key is None:
        key = jr.PRNGKey(0)

    def wrapped(*args, **kwargs):
        nonlocal key
        key, use_key = jr.split(key)
        return f(*args, key=use_key, **kwargs)

    return wrapped


def to_plain_data(value):
    """Convert configuration values to plain YAML-compatible scalars and containers."""
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_plain_data(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_dependency_versions():
    """Record the Python version and packages used by this project."""
    packages = (
        "laser", "cyclopts", "distrax", "flax", "gymnasium", "jax", "jaxlib", "jaxtyping",
        "numpy", "ogbench", "opencv-python", "optax", "pyyaml", "scipy", "tqdm", "wandb",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    restore_path = os.path.join(restore_path, f'params_{restore_epoch}.pkl')

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
