import os
from collections import deque
import math
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Any, Iterator, Sized
from abc import ABC, abstractmethod

import psutil
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
from loguru import logger

from episode import Episode

Batch = Dict[str, torch.Tensor]


class EpisodesDataset:
    def __init__(self, max_num_episodes: Optional[int] = None, name: Optional[str] = None) -> None:
        self.max_num_episodes = max_num_episodes
        self.name = name if name is not None else 'dataset'
        self.num_seen_episodes = 0
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def __len__(self) -> int:
        return len(self.episodes)

    def clear(self) -> None:
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()

    def add_episode(self, episode: Episode) -> int:
        if self.max_num_episodes is not None and len(self.episodes) == self.max_num_episodes:
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        return self.episodes[queue_idx]

    def update_episode(self, episode_id: int, new_episode: Episode) -> None:
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        merged_episode = self.episodes[queue_idx].merge(new_episode)
        self.episodes[queue_idx] = merged_episode
        self.newly_modified_episodes.add(episode_id)

    def _popleft(self) -> Episode:
        id_to_delete = [k for k, v in self.episode_id_to_queue_idx.items() if v == 0]
        assert len(id_to_delete) == 1
        self.newly_deleted_episodes.add(id_to_delete[0])
        self.episode_id_to_queue_idx = {k: v - 1 for k, v in self.episode_id_to_queue_idx.items() if v > 0}
        return self.episodes.popleft()

    def _append_new_episode(self, episode):
        episode_id = self.num_seen_episodes
        self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
        self.episodes.append(episode)
        self.num_seen_episodes += 1
        self.newly_modified_episodes.add(episode_id)
        return episode_id

    def sample_batch(self, batch_num_samples: int, sequence_length: int, weights: Optional[Tuple[float]] = None, sample_from_start: bool = True, context_len: int = 1) -> Batch:
        return self._collate_episodes_segments(self._sample_episodes_segments(batch_num_samples, sequence_length, weights, sample_from_start, context_len=context_len))

    def _sample_episodes_segments(self, batch_num_samples: int, sequence_length: int, weights: Optional[Tuple[float]], sample_from_start: bool, context_len: int = 1) -> List[Episode]:
        weights = np.array([max(0, len(e) - (context_len + 1)) for e in self.episodes])
        weights = weights / np.linalg.norm(weights)

        sampled_episodes = random.choices(self.episodes, k=batch_num_samples, weights=weights)

        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            if sample_from_start:
                start = random.randint(0, len(sampled_episode) - 1 - context_len)
                stop = start + sequence_length
            else:
                stop = random.randint(context_len + 1, len(sampled_episode))
                start = stop - sequence_length
            sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))
            assert len(sampled_episodes_segments[-1]) == sequence_length
        return sampled_episodes_segments

    def _collate_episodes_segments(self, episodes_segments: List[Episode]) -> Batch:
        episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        batch = {}
        for k in episodes_segments[0]:
            batch[k] = torch.stack([e_s[k] for e_s in episodes_segments])
        batch['observations'] = batch['observations'].float() / 255.0  # int8 to float and scale
        return batch

    def traverse(self, batch_num_samples: int, chunk_size: int, pad_dir: str = 'right'):
        for episode in self.episodes:
            if pad_dir == 'right':
                chunks = [episode.segment(start=i * chunk_size, stop=(i + 1) * chunk_size, should_pad=True)
                          for i in range(math.ceil(len(episode) / chunk_size))]
            else:
                assert pad_dir == 'left'
                chunks = [episode.segment(start=min((i + 1) * chunk_size, len(episode)) - chunk_size,
                                          stop=min((i + 1) * chunk_size, len(episode)),
                                          should_pad=True)
                          for i in range(math.ceil(len(episode) / chunk_size))]
            batches = [chunks[i * batch_num_samples: (i + 1) * batch_num_samples] for i in range(math.ceil(len(chunks) / batch_num_samples))]
            for b in batches:
                yield self._collate_episodes_segments(b)

    def update_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir()
        for episode_id in self.newly_modified_episodes:
            episode = self.get_episode(episode_id)
            episode.save(directory / f'{episode_id}.pt')
        for episode_id in self.newly_deleted_episodes:
            (directory / f'{episode_id}.pt').unlink()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def load_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir() and len(self.episodes) == 0
        episode_ids = sorted([int(p.stem) for p in directory.iterdir()])
        self.num_seen_episodes = episode_ids[-1] + 1
        for episode_id in episode_ids:
            episode = Episode(**torch.load(directory / f'{episode_id}.pt'))
            self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
            self.episodes.append(episode)


class EpisodesDatasetRamMonitoring(EpisodesDataset):
    """
    Prevent episode dataset from going out of RAM.
    Warning: % looks at system wide RAM usage while G looks only at process RAM usage.
    """
    def __init__(self, max_ram_usage: str, name: Optional[str] = None) -> None:
        super().__init__(max_num_episodes=None, name=name)
        self.max_ram_usage = max_ram_usage
        self.num_steps = 0
        self.max_num_steps = None

        max_ram_usage = str(max_ram_usage)
        if max_ram_usage.endswith('%'):
            m = int(max_ram_usage.split('%')[0])
            assert 0 < m < 100
            self.check_ram_usage = lambda: psutil.virtual_memory().percent > m
        else:
            assert max_ram_usage.endswith('G')
            m = float(max_ram_usage.split('G')[0])
            self.check_ram_usage = lambda: psutil.Process().memory_info()[0] / 2 ** 30 > m

    def clear(self) -> None:
        super().clear()
        self.num_steps = 0

    def add_episode(self, episode: Episode) -> int:
        if self.max_num_steps is None and self.check_ram_usage():
            self.max_num_steps = self.num_steps
            logger.info(f"Dataset RAM Threshold Reached! ({self.max_ram_usage})")
        self.num_steps += len(episode)
        while (self.max_num_steps is not None) and (self.num_steps > self.max_num_steps):
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def _popleft(self) -> Episode:
        episode = super()._popleft()
        self.num_steps -= len(episode)
        return episode


class EpisodesDatasetManager:
    def __init__(self, episodes_dir: Path, max_num_episodes: Optional[int] = None, name: Optional[str] = None,
                 sequence_length: int = 20, padding_strategy: str = 'right') -> None:
        self.episodes_dir = episodes_dir
        self.max_num_episodes = max_num_episodes
        self.name = name if name is not None else 'dataset'

        self.padding_strategy = padding_strategy
        self.sequence_length = sequence_length
        self._dataset_mngr = DirDEM(episodes_dir)
        assert self._dataset_mngr.__len__() == 0

    def __len__(self) -> int:
        return self._dataset_mngr.__len__()

    def clear(self) -> None:
        for ep_path in self._dataset_mngr.episodes_paths:
            assert ep_path.exists()
            os.remove(ep_path)
        self._dataset_mngr.refresh()

    def add_episode(self, episode: Episode) -> int:
        episode_id = self._append_new_episode(episode)
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        assert episode_id < self._dataset_mngr.__len__()
        assert self._dataset_mngr.get_episode_path(episode_id).name == self._get_episode_filename(episode_id)
        return self._dataset_mngr.get_episode(episode_id)

    def _get_episode_filename(self, episode_id: int) -> str:
        return f'{episode_id}.pt'

    def update_episode(self, episode_id: int, new_episode: Episode) -> None:
        assert episode_id == self._dataset_mngr.__len__() - 1
        merged_episode = self._dataset_mngr.get_episode(episode_id).merge(new_episode)
        episode_filename = self._get_episode_filename(episode_id)
        merged_episode.save(self._dataset_mngr.data_dir / episode_filename)
        self._dataset_mngr.update_episode(episode_filename)

    def _append_new_episode(self, episode):
        episode_id = self._dataset_mngr.__len__()
        episode_filename = self._get_episode_filename(episode_id)
        episode.save(self._dataset_mngr.data_dir / episode_filename)
        self._dataset_mngr.add_episode(episode_filename)
        return episode_id

    def get_iterable(self, batch_size: int, shuffle: bool = True):
        dataset = OfflineExperienceDataset(
            self._dataset_mngr,
            sequence_length=self.sequence_length,
            padding_strategy=self.padding_strategy
        )
        dataset_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset_iter


class DatasetEpisodesManager(ABC):
    def __init__(self, context_length: int = 1):
        self.context_length = context_length
        self.trajectory_lengths, self.trajectory_n_samples, self.trajectory_first_idx = self._extract_metadata()
        self._n_total_samples = sum(self.trajectory_n_samples)

    @abstractmethod
    def get_episode(self, episode_index: int):
        pass

    def get_num_episodes(self):
        return len(self.trajectory_first_idx)

    def get_episode_first_sample_index(self, episode_index: int):
        return self.trajectory_first_idx[episode_index]

    def get_episode_num_samples(self, episode_index: int):
        return self.trajectory_n_samples[episode_index]

    @property
    def samples_count(self):
        return self._n_total_samples

    def __len__(self):
        return self.samples_count

    def get_episode_index(self, sample_index):
        # perform binary search to find the index within trajectory_first_idx
        n = self.get_num_episodes()
        l, r = 0, n-1
        while l <= r:
            m = (l+r)//2
            if self.get_episode_first_sample_index(m) > sample_index:
                r = m - 1
            else:
                if m+1 < n:
                    if self.get_episode_first_sample_index(m+1) > sample_index:
                        return m
                    else:
                        l = m+1
                else:
                    assert m == n-1
                    return m

    @abstractmethod
    def _get_episodes_lengths(self) -> tuple[int, ...]:
        pass

    def _extract_metadata(self):
        # each trajectory .pt file is a dict with keys 'observations', 'actions', 'rewards', 'ends', 'mask_padding'.
        # observations are uint8 type, with shape (batch, 3, 64, 64)
        # actions, rewards, ends, mask_padding shape: (batch,)

        # extract the number of trajectories and their lengths:
        trajectory_lengths = []
        trajectory_n_samples = []
        trajectory_first_idx = []
        for episode_length in self._get_episodes_lengths():
            trajectory_lengths.append(episode_length)

            if len(trajectory_first_idx) == 0:
                first_idx = 0
            else:
                first_idx = trajectory_first_idx[-1] + trajectory_n_samples[-1]
            trajectory_first_idx.append(first_idx)

            # We don't need the last/first (depending on the padding direction) example where
            # there's only padding and a single observation.
            trajectory_n_samples.append(episode_length - self.context_length)
        # Compute the number of samples in each trajectory
        # (each sample is a fixed length sequence of (obs, action, reward) tuples):
        return trajectory_lengths, trajectory_n_samples, trajectory_first_idx


class DirDEM(DatasetEpisodesManager):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._trajectory_files = [fname for fname in sorted(self.data_dir.iterdir()) if fname.suffix == '.pt']

        super().__init__()

        logger.info(f"Got {len(self._trajectory_files)} trajectories, and {self._n_total_samples} total samples")

    def _get_episodes_lengths(self) -> tuple[int, ...]:
        return tuple([torch.load(self.data_dir / trajectory_file)['observations'].shape[0]
                      for trajectory_file in self._trajectory_files])

    def get_episode(self, episode_index: int):
        trajectory_file = self._trajectory_files[episode_index]
        trajectory = torch.load(self.data_dir / trajectory_file)
        return Episode(**trajectory)

    def add_episode(self, episode_filename: str):
        episode_path = self.data_dir / episode_filename
        assert episode_path.exists()
        self._trajectory_files.append(episode_path.relative_to(self.data_dir))
        trajectory_length = torch.load(episode_path)['observations'].shape[0]
        first_idx = self.trajectory_first_idx[-1] + self.trajectory_n_samples[-1]

        self.trajectory_lengths.append(trajectory_length)
        self.trajectory_n_samples.append(trajectory_length - 1)
        self.trajectory_first_idx.append(first_idx)
        self._n_total_samples += self.trajectory_n_samples[-1]

    def update_episode(self, episode_filename: str):
        assert episode_filename == self._trajectory_files[-1].name

        trajectory_length = torch.load(self.data_dir / episode_filename)['observations'].shape[0]

        self.trajectory_lengths[-1] = trajectory_length
        n_samples = trajectory_length - 1
        samples_diff = n_samples - self.trajectory_n_samples[-1]
        self.trajectory_n_samples[-1] = n_samples
        self._n_total_samples += samples_diff

    def refresh(self):
        self._trajectory_files = [fname for fname in sorted(self.data_dir.iterdir()) if fname.suffix == '.pt']
        self.trajectory_lengths, self.trajectory_n_samples, self.trajectory_first_idx = self._extract_metadata()
        self._n_total_samples = sum(self.trajectory_n_samples)

    @property
    def episodes_paths(self):
        return [fname.absolute() for fname in sorted(self.data_dir.iterdir()) if fname.suffix == '.pt']

    def get_episode_path(self, episode_index: int):
        return self._trajectory_files[episode_index].absolute()


class RAMDEM(DatasetEpisodesManager):
    def __init__(self, dataset: EpisodesDataset, context_length: int = 1):
        self.dataset = dataset
        super().__init__(context_length=context_length)

    def _get_episodes_lengths(self) -> tuple[int, ...]:
        return tuple([episode.observations.shape[0]
                      for episode in self.dataset.episodes])

    def get_episode(self, episode_index: int):
        return self.dataset.episodes[episode_index]


class OfflineExperienceDataset(Dataset):

    def __init__(self, episodes_manager: DatasetEpisodesManager, sequence_length: int, padding_strategy: str):
        self.padding_strategy = padding_strategy
        self.sequence_length = sequence_length
        self.episodes_manager = episodes_manager

    def __len__(self):
        return self.episodes_manager.samples_count

    def __getitem__(self, sample_index) -> T_co:
        file_idx = self.episodes_manager.get_episode_index(sample_index)
        episode = self.episodes_manager.get_episode(file_idx)

        sample_index_in_file = sample_index - self.episodes_manager.get_episode_first_sample_index(file_idx)
        if self.padding_strategy == 'right':
            start = sample_index_in_file
            stop = start + self.sequence_length
        else:
            assert self.padding_strategy == 'left'
            stop = self.episodes_manager.context_length + 1 + sample_index_in_file
            start = stop - self.sequence_length

        example = episode.segment(start, stop, should_pad=True).__dict__
        assert (example['mask_padding'].sum() > self.episodes_manager.context_length), f"Failed with sample index {sample_index} (idx in episode {sample_index_in_file}) ep len: {len(episode)}"
        example['observations'] = example['observations'].float() / 255.0
        return example


def get_dataloader(
        dataset: EpisodesDataset,
        context_length: int,
        sequence_length: int,
        batch_size: int,
        shuffle: bool,
        padding_strategy: str
):
    manager = RAMDEM(dataset, context_length=context_length)
    dataset = OfflineExperienceDataset(
        manager,
        sequence_length=sequence_length,
        padding_strategy=padding_strategy
    )
    dataset_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, persistent_workers=True)

    return dataset_iter
