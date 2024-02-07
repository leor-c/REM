from collections import deque
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
import torchvision

from utils import sample_categorical


def np_obs_to_tensor(obs, device):
    return torchvision.transforms.functional.to_tensor(obs).to(device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]


def np_obs_to_tokens_tensor(obs, device, tokenizer):
    return tokenizer.encode(np_obs_to_tensor(obs, device), should_preprocess=True).tokens


class RetNetWorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.incremental_state = None
        self._num_observations_tokens = world_model.config.tokens_per_block-1
        self.last_obs_tokens = None
        self.keys_values_wm = None
        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = self.env.reset()
        obs_tokens = np_obs_to_tokens_tensor(obs, self.device, self.tokenizer)
        ctx = [obs_tokens]
        for i in range(self.world_model.context_length-1):
            action = torch.zeros(1, 1, device=self.device).long()  # noop
            obs, reward, done, info = self.env.step(int(action[0].item()))
            ctx.append(action)
            ctx.append(np_obs_to_tokens_tensor(obs, self.device, self.tokenizer))
        ctx = torch.cat(ctx, dim=1)
        return self.reset_from_initial_observations(ctx)

    @torch.no_grad()
    def reset_from_initial_observations(self, ctx_tokens, return_tokens: bool = False) -> torch.FloatTensor:
        self.refresh_state_with_initial_obs_tokens(ctx_tokens)
        self.last_obs_tokens = ctx_tokens[:, -self._num_observations_tokens:]

        return self.decode_obs_tokens() if not return_tokens else ctx_tokens[:, -self._num_observations_tokens:]

    def query_world_model(self, tokens):
        outputs_wm = self.world_model(tokens, self.incremental_state)

        return outputs_wm

    @torch.no_grad()
    def refresh_state_with_initial_obs_tokens(self, ctx_tokens: torch.LongTensor) -> torch.FloatTensor:
        self.incremental_state = self.world_model.get_empty_state()
        outputs_wm = self.world_model(ctx_tokens, self.incremental_state)

        return outputs_wm  # (B, K, E)

    @torch.no_grad()
    def _compute_reward_and_done(self, outputs_wm: torch.Tensor):
        rewards_logits = self.world_model.head_rewards(outputs_wm)
        dones_logits = self.world_model.head_ends(outputs_wm)
        reward = Categorical(logits=rewards_logits).sample().float().cpu().numpy().reshape(-1) - 1  # (B,)
        done = Categorical(logits=dones_logits).sample().cpu().numpy().astype(bool).reshape(-1)  # (B,)

        return reward, done

    @torch.no_grad()
    def _compute_next_token(self, outputs_wm: torch.Tensor):
        obs_token_logits = self.world_model.head_observations(outputs_wm)
        return Categorical(logits=obs_token_logits).sample()

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True,
             return_tokens: bool = False) -> None:
        assert (self.keys_values_wm is not None or self.incremental_state is not None) and self.num_observations_tokens is not None

        action_token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long, device=self.device)
        action_token = action_token.reshape(-1, 1)  # (B, 1)
        if 'retnet' in type(self.world_model).__name__.lower():
            action_token = action_token + self.tokenizer.vocab_size
        outputs_wm = self.query_world_model(action_token)
        reward, done = self._compute_reward_and_done(outputs_wm)

        if not should_predict_next_obs:
            return None, reward, done, None

        self.last_obs_tokens = self._compute_next_obs_tokens(outputs_wm)

        obs = self.decode_obs_tokens() if not return_tokens else self.last_obs_tokens.clone()
        return obs, reward, done, None

    def _compute_next_obs_tokens(self, last_wm_output: torch.Tensor):
        output_sequence, obs_tokens = [], []

        token = self._compute_next_token(last_wm_output)

        for k in range(self.num_observations_tokens):  # assumption that there is only one action token (first token).
            obs_tokens.append(token)
            outputs_wm = self.query_world_model(token)

            token = self._compute_next_token(outputs_wm)

        return torch.cat(obs_tokens, dim=1)  # (B, K)

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> Tensor:
        embedded_tokens = self.tokenizer.embedding(self.last_obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.last_obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]


class POPRetNetWorldModelEnv(RetNetWorldModelEnv):

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device],
                 env: Optional[gym.Env] = None) -> None:
        super().__init__(tokenizer, world_model, device, env)
        self.prior_context = None

    @torch.no_grad()
    def refresh_state_with_initial_obs_tokens(self, ctx_tokens: torch.LongTensor):
        n, num_ctx_tokens = ctx_tokens.shape
        assert (num_ctx_tokens + 1) % self.world_model.config.tokens_per_block == 0

        self.incremental_state = {}
        self.prior_context = ctx_tokens

    def query_world_model(self, tokens):
        assert tokens.dim() == 2
        return self.world_model(tokens, incremental_state=self.incremental_state, tokenizer=self.tokenizer)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True,
             return_tokens: bool = False) -> None:
        assert (
                (self.keys_values_wm is not None or self.incremental_state is not None)
                and
                self.num_observations_tokens is not None
        )

        action_token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action,
                                                                                                     dtype=torch.long,
                                                                                                     device=self.device)
        action_token = action_token.reshape(-1, 1)  # (B, 1)
        action_token = action_token + self.tokenizer.vocab_size

        if self.prior_context is not None:
            tokens = torch.cat([self.prior_context, action_token], dim=1)
        else:
            tokens = action_token
        outputs_wm = self.query_world_model(tokens)

        reward, done = self._compute_reward_and_done(outputs_wm)

        if not should_predict_next_obs:
            return None, reward, done, None

        self.last_obs_tokens = self._compute_next_obs_tokens(outputs_wm)
        if self.world_model.uses_pop:
            self.prior_context = self.last_obs_tokens
        else:
            self.prior_context = None

        obs = self.decode_obs_tokens() if not return_tokens else self.last_obs_tokens.clone()
        return obs, reward, done, None

    def _compute_next_obs_tokens(self, last_wm_output: torch.Tensor):
        if self.world_model.uses_pop:
            if self.world_model.compute_states_parallel_inference:
                last_wm_output = last_wm_output[:, -self.world_model.config.tokens_per_block:-1]
                next_obs_logits = self.world_model.head_observations(last_wm_output)
                return sample_categorical(logits=next_obs_logits)
            else:
                preds = self.world_model.compute_next_obs_pred_latents(self.incremental_state)[0]
                next_obs_logits = self.world_model.head_observations(preds)
                return sample_categorical(logits=next_obs_logits)
        
        last_wm_output = last_wm_output[:, -1:]
        return super()._compute_next_obs_tokens(last_wm_output)

    @torch.no_grad()
    def _compute_reward_and_done(self, outputs_wm: torch.Tensor):
        if self.world_model.uses_pop:
            if self.world_model.compute_states_parallel_inference:
                k1 = self.world_model.config.tokens_per_block
                outputs_wm = outputs_wm[:, -k1-1:-k1]
            else:
                outputs_wm = outputs_wm[:, -1:]
            return super()._compute_reward_and_done(outputs_wm)

        outputs_wm = outputs_wm[:, -1:]
            
        return super()._compute_reward_and_done(outputs_wm)


class POPRetNetWMEnv4Play(POPRetNetWorldModelEnv):

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device],
                 env: Optional[gym.Env] = None) -> None:
        super().__init__(tokenizer, world_model, device, env)
        self.horizon = self.world_model.config.max_blocks
        self.context_length = self.world_model.context_length
        self.current_step = self.context_length
        self.next_context = deque([])

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        self.current_step = self.context_length
        self.next_context.clear()

        obs = self.env.reset()
        obs_tokens = np_obs_to_tokens_tensor(obs, self.device, self.tokenizer)
        self.next_context.append(obs_tokens)
        for i in range(self.world_model.context_length - 1):
            action = torch.zeros(1, 1, device=self.device).long()  # noop
            obs, reward, done, info = self.env.step(int(action[0].item()))
            self.next_context.append(action)
            self.next_context.append(np_obs_to_tokens_tensor(obs, self.device, self.tokenizer))
        ctx = torch.cat([v for v in self.next_context], dim=1)
        return self.reset_from_initial_observations(ctx)

    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True,
             return_tokens: bool = False) -> None:
        assert should_predict_next_obs
        res = super().step(action, should_predict_next_obs, return_tokens)
        obs, reward, done, info = self.env.step(action)

        action_token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action,
                                                                                                     dtype=torch.long,
                                                                                                     device=self.device)
        action_token = action_token.reshape(-1, 1)  # (B, 1)
        obs_tokens = np_obs_to_tokens_tensor(obs, self.device, self.tokenizer)
        self.next_context.append(action_token)
        self.next_context.append(obs_tokens)

        while len(self.next_context) > self.context_length * 2 - 1:
            self.next_context.popleft()

        self.current_step += 1

        if self.current_step % self.horizon - self.context_length == 0:
            self.refresh_state_with_initial_obs_tokens(torch.cat([v for v in self.next_context], dim=1))


        return res[0], res[1], done, info



