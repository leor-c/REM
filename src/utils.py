import re
from abc import ABC, abstractmethod
from collections import OrderedDict
import cv2
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Categorical

from episode import Episode


def configure_optimizer(model, learning_rate, weight_decay):
    decay = set()
    no_decay = set()

    for pn, p in model.named_parameters():
        param_names_lst = pn.split(sep='.')
        layer_norm_pattern = '\.ln(\d+)|(_f)\.'

        if param_names_lst[-1] == 'bias':
            no_decay.add(pn)
        elif 'norm' in pn:
            no_decay.add(pn)
        elif 'embedding' in param_names_lst or 'embed' in pn or 'pos_emb' in pn:
            no_decay.add(pn)
        elif re.search(layer_norm_pattern, pn) is not None:
            no_decay.add(pn)
        else:
            assert param_names_lst[-1] in ['weight', 'freq'] or re.search("w_[a-z]$", param_names_lst[-1].lower()) is not None
            decay.add(pn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set! keys: {param_dict.keys() - union_params}"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class EpisodeDirManager:
    def __init__(self, episode_dir: Path, max_num_episodes: int, disable_saving: bool = False) -> None:
        self.disable_saving = disable_saving
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.disable_saving:
            return

        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        episode.save(self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            episode.save(self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')

class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()


def sample_categorical(logits: torch.Tensor, temp: float = 1.0) -> torch.LongTensor:
    # assert logits.dim() == 3  # (batch, seq len, k-options to sample)
    return Categorical(logits=logits / temp).sample().long()


class TrainerInfoHandler(ABC):
    @abstractmethod
    def signal_epoch_start(self):
        pass

    @abstractmethod
    def update_with_step_info(self, step_info: dict):
        pass

    @abstractmethod
    def get_epoch_info(self) -> dict:
        pass


class GradNormInfo:
    def __init__(self):
        self.epoch_grad_norms = []

    def reset(self):
        self.epoch_grad_norms.clear()

    def __call__(self, iter_grad_norm, *args, **kwargs):
        self.epoch_grad_norms.append(iter_grad_norm)

    def get_info(self):
        return {
            'max_grad_norm': np.max(self.epoch_grad_norms),
            'mean_grad_norm': np.mean(self.epoch_grad_norms),
            'min_grad_norm': np.min(self.epoch_grad_norms)
        }
