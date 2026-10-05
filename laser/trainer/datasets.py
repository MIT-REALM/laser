import jax.tree_util as jtu
import numpy as np

from flax.core.frozen_dict import FrozenDict
from cyclopts import Parameter
from dataclasses import dataclass


def get_size(data):
    """Return the size of the dataset."""
    sizes = jtu.tree_map(lambda arr: len(arr), data)
    return max(jtu.tree_leaves(sizes))


@Parameter(name="*", group="DatasetConfig")
@dataclass
class DatasetCfg:
    """Configuration for the dataset.

    Parameters
    ----------
    frame_stack : int | None
        Number of observations to stack; must be a positive integer when provided.
    """
    frame_stack: int | None = None

    def __post_init__(self):
        value = self.frame_stack
        if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"frame_stack must be a positive integer or None (got {value!r}).")


class Dataset(FrozenDict):

    @classmethod
    def create(cls, cfg: DatasetCfg, seed: int, freeze=True, **fields):
        data = fields
        assert 'observations' in data
        if freeze:
            jtu.tree_map(lambda arr: arr.setflags(write=False), data)
        rng = np.random.default_rng(seed)
        return cls(cfg, rng, data)

    def __init__(self, cfg: DatasetCfg, rng: np.random.Generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.cfg = cfg
        self.rng = rng

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs + 1])
        self.initial_locs = self.initial_locs[self.initial_locs < self.size]

    def get_random_indices(self, num_indices: int) -> np.ndarray:
        """Return `num_indices` random indices."""
        return self.rng.integers(0, self.size, size=num_indices)

    def sample(self, batch_size: int, indices=None) -> dict:
        """Sample a batch of transitions."""
        if indices is None:
            indices = self.get_random_indices(batch_size)
        return self.get_subset(indices)

    def get_subset(self, indices) -> dict:
        """Return transitions with the same observation stacking used during rollouts."""
        indices = np.asarray(indices)
        batch = jtu.tree_map(lambda arr: arr[indices], self._dict)
        if self.cfg.frame_stack is not None:
            initial_indices = self.initial_locs[np.searchsorted(self.initial_locs, indices, side='right') - 1]
            observations = []
            for offset in reversed(range(self.cfg.frame_stack)):
                # Pad episode starts with the initial observation.
                cur_indices = np.maximum(indices - offset, initial_indices)
                observations.append(self['observations'][cur_indices])
            next_observations = observations[1:] + [batch['next_observations']]
            batch['observations'] = np.concatenate(observations, axis=-1)
            batch['next_observations'] = np.concatenate(next_observations, axis=-1)
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        """Sample sequences with discounted returns and episode-boundary masks."""
        if not 1 <= sequence_length <= self.size:
            raise ValueError(f"sequence_length must be between 1 and {self.size} (got {sequence_length!r}).")
        idxs = self.rng.integers(self.size - sequence_length + 1, size=batch_size)
        data = self.get_subset(idxs)

        rewards = np.zeros(data['rewards'].shape + (sequence_length,), dtype=float)
        masks = np.ones(data['masks'].shape + (sequence_length,), dtype=float)
        valid = np.ones(data['masks'].shape + (sequence_length,), dtype=float)
        next_observations = np.zeros(
            data['observations'].shape[:-1] + (sequence_length, data['observations'].shape[-1]), dtype=float)
        actions = np.zeros(data['actions'].shape[:-1] + (sequence_length, data['actions'].shape[-1]), dtype=float)
        terminals = np.zeros(data['terminals'].shape + (sequence_length,), dtype=float)

        for i in range(sequence_length):
            step_data = data if i == 0 else self.get_subset(idxs + i)
            actions[..., i, :] = step_data['actions']
            if i == 0:
                rewards[..., i] = step_data['rewards']
                masks[..., i] = step_data['masks']
                terminals[..., i] = step_data['terminals']
                next_observations[..., i, :] = step_data['next_observations']
            else:
                valid[..., i] = 1.0 - terminals[..., i - 1]
                rewards[..., i] = rewards[..., i - 1] + step_data['rewards'] * discount ** i * valid[..., i]
                masks[..., i] = np.where(
                    valid[..., i], np.minimum(masks[..., i - 1], step_data['masks']), masks[..., i - 1])
                terminals[..., i] = np.maximum(terminals[..., i - 1], step_data['terminals'])
                # Hold the final next observation after the episode ends.
                next_observations[..., i, :] = np.where(
                    valid[..., i:i + 1], step_data['next_observations'], next_observations[..., i - 1, :])

        return dict(
            observations=data['observations'].copy(),
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
        )
