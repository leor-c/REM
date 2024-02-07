from dataclasses import dataclass
from typing import Any, Optional, Union
import sys

import loguru
from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import Batch
from envs.world_model_env import RetNetWorldModelEnv, POPRetNetWorldModelEnv
from models.tokenizer import Tokenizer
from models.world_model import RetNetWorldModel, POPRetNetWorldModel
from utils import compute_lambda_returns, LossWithIntermediateLosses


@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    means_values: torch.FloatTensor


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size, use_original_obs: bool = False, lstm_latent_dim: int = 512, name: str = "AC") -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm_dim = lstm_latent_dim
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None

        self.critic_linear = nn.Linear(lstm_latent_dim, 1)
        self.actor_linear = nn.Linear(lstm_latent_dim, act_vocab_size)

        loguru.logger.info("Initialized ActorCritic")

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        device = self.conv1.weight.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 5 and burnin_observations.size(0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
        assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
        x = inputs[mask_padding] if mask_padding is not None else inputs

        x = x.mul(2).sub(1)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)

        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))

        logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        means_values = rearrange(self.critic_linear(self.hx), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, means_values)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: RetNetWorldModel, imagine_horizon: int, gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        assert not self.use_original_obs
        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])
        log_probs = d.log_prob(outputs.actions[:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy), {}

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: RetNetWorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        if isinstance(world_model, POPRetNetWorldModel):
            return self.imagine_with_pop(batch, tokenizer, world_model, horizon, show_pbar)
        elif isinstance(world_model, RetNetWorldModel):
            return self.imagine_no_pop(batch, tokenizer, world_model, horizon, show_pbar)
        else:
            assert False, 'Model not supported'

    def imagine_no_pop(self, batch: Batch, tokenizer: Tokenizer, world_model: RetNetWorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -2:].all()
        device = initial_observations.device
        wm_env = RetNetWorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        encoded_obs = tokenizer.encode(initial_observations, should_preprocess=True)
        burnin_observations = torch.clamp(tokenizer.decode(encoded_obs.z_quantized[:, :-1], should_postprocess=True), 0, 1) if initial_observations.size(1) > 1 else None
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1])

        obs = wm_env.reset_from_initial_observations(encoded_obs.tokens[:, -world_model.config.tokens_per_block])
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):

            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            obs, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=(k < horizon - 1))

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )

    def imagine_with_pop(self, batch: Batch, tokenizer: Tokenizer, world_model: RetNetWorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device
        wm_env = POPRetNetWorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        encoded_obs = tokenizer.encode(initial_observations, should_preprocess=True)
        burnin_observations = torch.clamp(tokenizer.decode(encoded_obs.z_quantized[:, :-1], should_postprocess=True), 0,
                                          1) if initial_observations.size(1) > 1 else None
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1])

        ctx_len = world_model.context_length
        action_tokens = batch['actions'][:, -ctx_len:].unsqueeze(2)
        if 'retnet' in type(world_model).__name__.lower():
            action_tokens = action_tokens + tokenizer.vocab_size
        ctx = torch.cat([encoded_obs.tokens[:, -ctx_len:], action_tokens], dim=2).flatten(1, 2)[:, :-1]
        obs = wm_env.reset_from_initial_observations(ctx, return_tokens=False)
        effective_horizon = horizon - ctx_len + 1
        for k in tqdm(range(effective_horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):

            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            should_predict_next_obs = k < effective_horizon - 1
            obs, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=should_predict_next_obs, return_tokens=False)

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1),      # (B, T, C, H, W)
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )



class ActorCriticLS(nn.Module):
    """
    This version works in latent space. it receives the token codes of the observation frame (embed_dim, w, h)
    and applies a CNN + dense to map it to a latent vector which is the input to the LSTM.
    Importantly, the token codes are the learned codes of the tokenizer.
    """
    def __init__(self, act_vocab_size, token_embed_dim, tokens_per_obs: int, lstm_latent_dim: int = 512,
                 use_original_obs=False, name: str = "LSAC", n_layers: int = 2, include_action_inputs: bool = True, context_len: int = 2, rnn_type: str = 'lstm') -> None:
        super().__init__()
        self.name = name
        self.include_action_inputs = include_action_inputs
        self.use_original_obs = use_original_obs
        self.token_embed_dim = token_embed_dim
        self.tokens_per_obs = tokens_per_obs
        self.context_len = context_len

        self.lstm_dim = lstm_latent_dim

        self.n_layers = n_layers
        self.convs, self.dense = self._build_cnn()

        self.rnn_type = rnn_type
        self.rnn, self.rnn_state = self._build_model()

        self.critic_linear = nn.Linear(lstm_latent_dim, 1)
        self.actor_linear = nn.Linear(lstm_latent_dim, act_vocab_size)

        self.actions_embeddings = nn.Embedding(act_vocab_size, lstm_latent_dim)

        loguru.logger.info("Initialized ActorCriticLS")

    def _build_model(self):
        if self.rnn_type == 'lstm':
            lstm = nn.LSTMCell(self.lstm_dim, self.lstm_dim)
            lstm_state = (None, None)
            return lstm, lstm_state
        
        elif self.rnn_type == 'gru':
            gru = nn.GRUCell(self.lstm_dim, self.lstm_dim)
            gru_state = None
            return gru, gru_state

    def _build_cnn(self):
        token_embed_dim = self.token_embed_dim
        convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(token_embed_dim // (2 ** i), token_embed_dim // (2 ** (i + 1)), 3, 1, 1),
                nn.SiLU(),
            )
            for i in range(self.n_layers)
        ])

        in_features = self.tokens_per_obs * token_embed_dim // (2 ** self.n_layers)
        dense = nn.Sequential(
            nn.Linear(in_features, self.lstm_dim),
            nn.SiLU(),
        )
        return convs, dense

    def __repr__(self) -> str:
        return self.name

    def clear(self) -> None:
        if self.rnn_type == 'lstm':
            self.rnn_state = (None, None)
        elif self.rnn_type == 'gru':
            self.rnn_state = None

    def get_zero_rnn_state(self, n, device, rnn_type: str = None):
        if rnn_type is None:
            rnn_type = self.rnn_type

        if rnn_type == 'lstm':
            return torch.zeros(n, self.lstm_dim, device=device), torch.zeros(n, self.lstm_dim, device=device)
        elif rnn_type == 'gru':
            return torch.zeros(n, self.lstm_dim, device=device)
        else:
            assert False, f"rnn type '{rnn_type}' not supported"

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None, actions = None) -> None:
        device = self.convs[0][0].weight.device
        self.rnn_state = self.get_zero_rnn_state(n, device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 5 and burnin_observations.size(0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])
                if (i+1) % self.tokens_per_obs == 0 and self.include_action_inputs:
                    self(mask_padding=mask_padding[:, i], action=actions[:, i])

    def prune(self, mask: np.ndarray) -> None:
        if self.rnn_type == 'lstm':
            hx, cx = self.rnn_state
            hx = hx[mask]
            cx = cx[mask]
            self.rnn_state = (hx, cx)
        elif self.rnn_type == 'gru':
            self.rnn_state = self.rnn_state[mask]

    def rnn_forward(self, x, mask_padding):
        if mask_padding is None:
            self.rnn_state = self.rnn(x, self.rnn_state)
        else:
            if self.rnn_type == 'lstm':
                hx, cx = self.rnn_state
                hx[mask_padding], cx[mask_padding] = self.rnn(x, (hx[mask_padding], cx[mask_padding]))
                self.rnn_state = (hx, cx)
            
            elif self.rnn_type == 'gru':
                self.rnn_state[mask_padding] = self.rnn(x, self.rnn_state[mask_padding])
            else:
                assert False, f"rnn type '{self.rnn_type}' not supported"

    def get_rnn_output(self):
        if self.rnn_type == 'lstm':
            return self.rnn_state[0]
        elif self.rnn_type == 'gru':
            return self.rnn_state
        else:
            assert False, f"rnn type '{self.rnn_type}' not supported"

    def forward(self, inputs: torch.FloatTensor=None, mask_padding: Optional[torch.BoolTensor] = None, action=None) -> ActorCriticOutput:
        assert (inputs is None) != (action is None)  # XOR
        if inputs is not None:
            assert inputs.ndim == 4 and inputs.shape[1] == self.token_embed_dim
            # assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
            assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
            x = inputs[mask_padding] if mask_padding is not None else inputs

            for i in range(self.n_layers):
                x = self.convs[i](x)

            x = x.flatten(1)

            x = self.dense(x)
        else:
            assert self.include_action_inputs
            assert action is not None
            action = action[mask_padding] if mask_padding is not None else action
            assert action.dim() == 1
            x = self.actions_embeddings(action.flatten())

        self.rnn_forward(x, mask_padding)
        
        x = self.get_rnn_output()
        logits_actions = rearrange(self.actor_linear(x), 'b a -> b 1 a')
        means_values = rearrange(self.critic_linear(x), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, means_values)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: RetNetWorldModel, imagine_horizon: int,
                     gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        assert not self.use_original_obs
        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])
        log_probs = d.log_prob(outputs.actions[:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns, reduction='none').mean(dim=1)
        per_sample_loss = loss_values.detach()
        loss_values = loss_values.mean()

        info = {'per_sample_loss': per_sample_loss}

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy), info

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: RetNetWorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device
        wm_env = POPRetNetWorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        encoded_obs = tokenizer.encode(initial_observations, should_preprocess=True)
        burnin_observations = encoded_obs.z_quantized[:, :-1] if initial_observations.size(1) > 1 else None
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1], actions=batch['actions'][:, :-1])

        ctx_len = self.context_len
        action_tokens = batch['actions'][:, -ctx_len:].unsqueeze(2)
        if 'retnet' in type(world_model).__name__.lower():
            action_tokens = action_tokens + tokenizer.vocab_size
        ctx = torch.cat([encoded_obs.tokens[:, -ctx_len:], action_tokens], dim=2).flatten(1, 2)[:, :-1]
        obs_tokens = wm_env.reset_from_initial_observations(ctx, return_tokens=True)
        obs_codes = tokenizer.to_codes(obs_tokens)
        effective_horizon = horizon - ctx_len + 1
        for k in tqdm(range(effective_horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):

            all_observations.append(obs_codes)

            outputs_ac = self(inputs=obs_codes)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            if self.include_action_inputs:
                self(action=action_token.flatten())
            should_predict_next_obs = k < effective_horizon - 1
            obs_tokens, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=should_predict_next_obs, return_tokens=True)
            obs_codes = tokenizer.to_codes(obs_tokens) if should_predict_next_obs else None

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1),      # (B, T, C, H, W)
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )
