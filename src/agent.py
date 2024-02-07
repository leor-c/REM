from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.actor_critic import ActorCritic, ActorCriticLS
from models.tokenizer import Tokenizer
from models.world_model import RetNetWorldModel
from utils import extract_state_dict


class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: RetNetWorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token

    def reset_actor_critic(self, n, burnin_observations, mask_padding, actions=None):
        b_o = burnin_observations
        if burnin_observations is not None:
            b_o = torch.clamp(self.tokenizer.encode_decode(burnin_observations, should_preprocess=True, should_postprocess=True), 0, 1)
        return self.actor_critic.reset(n=n, burnin_observations=b_o, mask_padding=mask_padding)


class AgentLS(Agent):
    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        assert isinstance(self.actor_critic, ActorCriticLS)
        input_ac = self.tokenizer.encode(obs, should_preprocess=True).z_quantized
        logits_actions = self.actor_critic(inputs=input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        if self.actor_critic.include_action_inputs:
            self.actor_critic(action=act_token)
        return act_token

    def reset_actor_critic(self, n, burnin_observations, mask_padding, actions=None):
        assert isinstance(self.actor_critic, ActorCriticLS)
        b_o = burnin_observations
        if burnin_observations is not None:
            b_o = self.tokenizer.encode(burnin_observations, should_preprocess=True).z_quantized
        return self.actor_critic.reset(n=n, burnin_observations=b_o, mask_padding=mask_padding, actions=actions)
