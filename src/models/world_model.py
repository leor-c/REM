from dataclasses import dataclass
from math import ceil
from typing import Any, Optional, Tuple

from loguru import logger
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from yet_another_retnet.retnet import RetNetDecoder, RetNetDecoderLayer

from dataset import Batch

from .pop_retnet import POPRetNetDecoderLayer, POPRetNetDecoder
from .tokenizer import Tokenizer
from utils import LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


@dataclass
class RetNetConfig:
    tokens_per_block: int
    max_blocks: int

    num_layers: int
    num_heads: int
    embed_dim: int

    dropout: float

    blocks_per_chunk: int

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks

    @property
    def tokens_per_chunk(self):
        return self.blocks_per_chunk * self.tokens_per_block


class RetNetWorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: RetNetConfig, context_length: int = 2,
                 shared_embeddings: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config

        self.head_observations = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, obs_vocab_size)
        )

        self.head_rewards = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 3)
        )

        self.head_ends = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 2)
        )

        self.context_length = context_length
        self.config = config

        self.embedding = nn.Embedding(self.vocab_size, config.embed_dim)
        self.shared_embeddings = shared_embeddings
        self._model = self._build_model()
        self._model.out = nn.Identity()
        self._device = None
        logger.info(f"Initialized {self.__class__}. Shared embeddings: {self.shared_embeddings}")

    def __repr__(self):
        return "world_model"

    def _build_model(self):
        decoder_layer = RetNetDecoderLayer(
            self.config.embed_dim,
            self.config.num_heads,
            dropout=self.config.dropout,
            dim_feedforward=4 * self.config.embed_dim,
        )
        return RetNetDecoder(decoder_layer, self.config.num_layers)

    @property
    def vocab_size(self):
        return self.obs_vocab_size + self.act_vocab_size + 1  # + pad token (last)

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def get_empty_state(self):
        return {
            'state': [],
            'n': 0
        }

    def init_retnet_state(self, incremental_state: dict):
        assert len(incremental_state) == 0
        incremental_state['state'] = []
        incremental_state['n'] = 0

    def _embed_tokens(self, tokens, tokenizer: Optional[Tokenizer] = None):
        if self.shared_embeddings:
            tmp_combined_emb = nn.Embedding.from_pretrained(
                embeddings=torch.cat((tokenizer.embedding.weight.data, self.embedding.weight[self.obs_vocab_size:]),
                                     dim=0),
                freeze=False
            )
            tokens_emb = tmp_combined_emb(tokens)
        else:
            tokens_emb = self.embedding(tokens)
        return tokens_emb

    def forward(self, tokens: torch.LongTensor, incremental_state=None, tokenizer: Optional[Tokenizer] = None):
        num_steps = tokens.shape[1]

        assert tokens.dim() == 2
        tokens_emb = self._embed_tokens(tokens, tokenizer)
        if num_steps > self.config.tokens_per_chunk:
            return self._transparent_chunkwise_forward(tokens_emb, incremental_state)

        return self._forward_retnet(tokens_emb, incremental_state)

    def _forward_retnet(self, tokens_emb, incremental_state):
        assert tokens_emb.dim() == 3
        num_steps = tokens_emb.size(1)  # (B, T, e)
        assert num_steps <= self.config.tokens_per_chunk

        if incremental_state is None:
            x = self._model.forward_parallel(tokens_emb)  # parallel forward
        else:
            assert isinstance(incremental_state, dict)
            if len(incremental_state) == 0:
                self.init_retnet_state(incremental_state)

            if num_steps > 1:
                x, state = self._model.forward_chunkwise(tokens_emb, incremental_state['n'], incremental_state['state'])
                incremental_state['n'] += num_steps
            else:
                assert num_steps == 1
                x, state = self._model.forward_recurrent(tokens_emb[:, -1], incremental_state['n'], incremental_state['state'])
                x = rearrange(x, 'b e -> b 1 e')
                incremental_state["n"] += 1

            incremental_state["state"] = state

        return x

    def _transparent_chunkwise_forward(self, tokens_emb, incremental_state):
        assert tokens_emb.dim() == 3
        num_steps = tokens_emb.size(1)  # (B, T, e)
        assert num_steps > self.config.tokens_per_chunk

        if incremental_state is None:
            # pseudo parallel forward, performed with multiple chunkwise forward calls. Transparent to the user.
            incremental_state = {}
            self.init_retnet_state(incremental_state)

        outputs = []
        n: int = incremental_state['n']
        for i in range(n, n + num_steps, self.config.tokens_per_chunk):
            out, incremental_state['state'] = self._model.forward_chunkwise(
                tokens_emb[:, i:i+self.config.tokens_per_chunk],
                i,
                incremental_state['state']
            )
            incremental_state['n'] += out.shape[1]
            outputs.append(out)

        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def _get_tokens(self, batch, tokenizer):
        obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (BL, K)
        assert obs_tokens.dim() == 3 and obs_tokens.shape[2] == self.config.tokens_per_block-1
        act_tokens = rearrange(batch['actions'] + tokenizer.vocab_size, 'b l -> b l 1')
        return torch.cat((obs_tokens, act_tokens), dim=2)  # (B, L, K+1)

    def get_actions_mask(self, batch_size: int, seq_len: int, device):
        mask = torch.zeros(batch_size, seq_len, device=device, dtype=bool)
        tokens_per_obs = self.config.tokens_per_block - 1
        mask[:, tokens_per_obs::self.config.tokens_per_block] = 1
        return mask

    def get_non_action_mask(self, batch_size: int, seq_len: int, device):
        return torch.logical_not(self.get_actions_mask(batch_size, seq_len, device))

    def get_next_token_logits_and_labels(self, outputs: torch.Tensor, tokens, mask_padding: torch.Tensor):
        bsz, token_seq_len, emb_dim = outputs.shape
        non_action_mask = self.get_non_action_mask(bsz, token_seq_len, outputs.device)
        context_len = self.context_length
        non_action_mask[:, :self.config.tokens_per_block * context_len] = 0
        mask_p = mask_padding.unsqueeze(2).expand(-1, -1, self.config.tokens_per_block).flatten(1, 2)
        assert mask_p.shape == non_action_mask.shape, f"Incompatible shapes: {mask_p.shape} and {non_action_mask.shape}"

        labels_mask = torch.logical_and(non_action_mask, mask_p)

        labels = tokens[torch.where(labels_mask)]

        locs = torch.where(torch.flatten(labels_mask))[0] - 1
        relevant_elements_mask = torch.zeros_like(labels_mask).flatten()
        relevant_elements_mask[locs] = 1
        relevant_elements_mask = torch.reshape(relevant_elements_mask, labels_mask.shape)

        y = outputs[torch.where(relevant_elements_mask)]
        logits = self.head_observations(y)

        return logits, labels

    def get_rewards_ends_logits_and_labels(self, outputs: torch.Tensor, mask_padding: torch.Tensor, rewards, ends):
        action_tokens_logits = rearrange(outputs, 'b (t k1) e -> b t k1 e', k1=self.config.tokens_per_block)[:, :, -1]
        relevant_elements_mask = mask_padding.clone()
        relevant_elements_mask[:, :self.context_length] = 0
        y = action_tokens_logits[torch.where(relevant_elements_mask)]
        rewards_logits = self.head_rewards(y)
        ends_logits = self.head_ends(y)

        rewards_labels = (rewards[torch.where(relevant_elements_mask)].sign() + 1).long()
        ends_labels = ends[torch.where(relevant_elements_mask)]

        return rewards_logits, rewards_labels, ends_logits, ends_labels

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:

        tokens = self._get_tokens(batch, tokenizer) if 'tokens' not in batch else batch['tokens']
        tokens = rearrange(tokens, 'b t k1 -> b (t k1)')

        if self.shared_embeddings:
            outputs = self(tokens, tokenizer=tokenizer)
        else:
            outputs = self(tokens)

        next_token_logits, next_token_labels = self.get_next_token_logits_and_labels(outputs, tokens, batch['mask_padding'])

        rewards_logits, labels_rewards, ends_logits, labels_ends = self.get_rewards_ends_logits_and_labels(
            outputs,
            batch['mask_padding'],
            batch['rewards'],
            batch['ends']
        )

        loss_obs = F.cross_entropy(next_token_logits, next_token_labels)
        loss_rewards = F.cross_entropy(rewards_logits, labels_rewards)
        loss_ends = F.cross_entropy(ends_logits, labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends), {}

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        ignore = -100
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), ignore), 'b t k -> b (t k)')[:, 1:]
        seq_len = labels_observations.shape[1]
        num_labels_per_example = torch.sum(labels_observations >= 0, dim=1)
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, ignore).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, ignore)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1), num_labels_per_example, seq_len

    @property
    def uses_pop(self) -> bool:
        return False


class POPRetNetWorldModel(RetNetWorldModel):
    def __init__(
            self,
            obs_vocab_size: int,
            act_vocab_size: int,
            retnet: RetNetConfig,
            context_length: int = 2,
            compute_states_parallel: bool = True,
            shared_embeddings: bool = True,
            shared_prediction_token: bool = False,
            **kwargs
    ) -> None:
        super().__init__(obs_vocab_size, act_vocab_size, retnet, context_length)
        self.placeholder_embeddings = nn.Embedding(retnet.tokens_per_block, retnet.embed_dim)
        self.embedding = nn.Embedding(self.vocab_size, retnet.embed_dim)
        self.compute_states_parallel = compute_states_parallel
        self.compute_states_parallel_inference = False
        self.shared_embeddings = shared_embeddings
        self.pred_tokens_version = 'shared' if shared_prediction_token else 'per-token'

    def _build_model(self):
        decoder_layer = POPRetNetDecoderLayer(
            self.config.embed_dim,
            self.config.num_heads,
            dropout=self.config.dropout,
            dim_feedforward=4*self.config.embed_dim,
        )
        return POPRetNetDecoder(decoder_layer, self.config.num_layers)

    def _embed_tokens_shared(self, tokens, tokenizer):
        if tokens.dim() == 2:
            tokens = rearrange(tokens, 'b (t k1) -> b t k1', k1=self.config.tokens_per_block)

        assert tokens.dim() == 3
        obs_tokens_emb = tokenizer.to_codes(tokens.flatten(0, 1)[:, :-1])
        obs_tokens_emb = rearrange(obs_tokens_emb, '(b t) e h w -> b t (h w) e', t=tokens.shape[1])
        action_tokens_emb = self.embedding(tokens[:, :, -1:])
        assert action_tokens_emb.dim() == 4 and action_tokens_emb.shape[2] == 1
        tokens_emb = torch.cat([obs_tokens_emb, action_tokens_emb], dim=2)
        return tokens_emb

    def _embed_tokens(self, tokens, tokenizer: Optional[Tokenizer] = None):
        if tokens.dim() == 2:
            tokens = rearrange(tokens, 'b (t k1) -> b t k1', k1=self.config.tokens_per_block)

        assert tokens.dim() == 3 and tokens.shape[2] == self.config.tokens_per_block

        if self.shared_embeddings:
            assert tokenizer is not None
            tokens_emb = self._embed_tokens_shared(tokens, tokenizer)

            if tokens_emb.dim() == 3:
                tokens_emb = rearrange(tokens_emb, 'b (t k1) e -> b t k1 e', k1=self.config.tokens_per_block)

            assert tokens_emb.dim() == 4 and tokens_emb.shape[2] == self.config.tokens_per_block

        else:
            tokens_emb = self.embedding(tokens)

        return tokens_emb

    def forward(
            self,
            tokens: torch.LongTensor,
            incremental_state=None,
            compute_next_state_latents: bool = False,
            tokenizer: Optional[Tokenizer] = None
    ):

        initial_state = incremental_state
        if initial_state is None:
            initial_state = self.get_empty_state()

        tokens_emb = self._embed_tokens(tokens, tokenizer)

        assert isinstance(self._model, POPRetNetDecoder)

        if compute_next_state_latents:
            if self.compute_states_parallel:
                return self._compute_train_forward_parallel(tokens_emb, initial_state)
            else:
                return self._compute_train_forward_sequential(tokens_emb, initial_state)
        else:
            assert tokens_emb.shape[1] < self.config.max_blocks
            bsz, num_steps = tokens_emb.shape[:2]
            pred_emb = self._get_prediction_tokens_embeddings(bsz, 1, tokens_emb.device)
            tokens_emb = rearrange(tokens_emb, 'b t k1 e -> b (t k1) e')
            tokens_emb2 = torch.cat([tokens_emb, pred_emb], dim=1)
            if incremental_state is None:
                return self._model.forward_parallel(tokens_emb)
            else:
                if len(incremental_state) == 0:
                    self.init_retnet_state(incremental_state)

                if tokens_emb.shape[1] == 1:
                    assert False, 'unsupported length!'
                else:
                    if not self.compute_states_parallel_inference:
                        outs, incremental_state['state'] = self._model.forward_chunkwise(tokens_emb, incremental_state['n'], incremental_state['state'])
                    else:
                        outs, state_per_frame = self._model.forward_chunkwise_per_block_state(tokens_emb2, incremental_state['n'], tokens_per_block=self.config.tokens_per_block, prev_states=incremental_state['state'])
                        incremental_state['state'] = [s[-2] for s in state_per_frame]
                incremental_state['n'] += tokens_emb.shape[1]
                return outs

    def _compute_train_forward_sequential(self, tokens_emb, initial_state: Optional[dict]):
        bsz, num_steps = tokens_emb.shape[:2]
        pred_tokens_emb = self._get_prediction_tokens_embeddings(bsz, 1, tokens_emb.device)[:, :-1]
        outputs = []

        for t in range(num_steps):
            pred_outs, _ = self._model.forward_chunkwise(pred_tokens_emb, initial_state['n'], initial_state['state'])
            outputs.append(pred_outs)

            tokens_emb_t = tokens_emb[:, t]
            step_outs, initial_state['state'] = self._model.forward_chunkwise(tokens_emb_t, initial_state['n'], initial_state['state'])

            initial_state['n'] += tokens_emb_t.shape[1]
            outputs.append(step_outs[:, -1:])
            assert tokens_emb_t.shape[1] == self.config.tokens_per_block

        return torch.cat(outputs, dim=1)

    def _compute_train_forward_parallel(self, tokens_emb, initial_state: Optional[dict]):
        bsz, num_steps = tokens_emb.shape[:2]

        blocks_per_chunk = self.config.blocks_per_chunk
        n_chunks = ceil(num_steps / blocks_per_chunk)
        outs = []
        for i in range(n_chunks):
            start, stop = i * blocks_per_chunk, min((i+1) * blocks_per_chunk, num_steps)
            tokens_emb_i = tokens_emb[:, start:stop].flatten(1, 2)
            outs1, state_per_block = self._model.forward_chunkwise_per_block_state(
                tokens_emb_i,
                start_idx=initial_state['n'],
                tokens_per_block=self.config.tokens_per_block,
                prev_states=initial_state['state']
            )
            
            pred_tokens_emb = self._get_prediction_tokens_embeddings(bsz, stop - start, tokens_emb.device)

            if i == 0:
                s0 = torch.zeros_like(state_per_block[0][0:1])
                shifted_states = [torch.cat([s0, s_i[:-1]], dim=0) for s_i in state_per_block]
            else:
                shifted_states = [torch.cat([s0.unsqueeze(0), s_i[:-1]], dim=0) for s0, s_i in zip(initial_state['state'], state_per_block)]

            outs2, _ = self._model.forward_chunkwise_from_per_block_states(
                pred_tokens_emb,
                initial_state['n'],
                tokens_per_block=self.config.tokens_per_block,
                per_block_states=shifted_states,
                compute_state=False
            )
            outs2 = rearrange(outs2, 'b (t k1) d -> b t k1 d', k1=self.config.tokens_per_block)
            outs1 = rearrange(outs1, 'b (t k1) d -> b t k1 d', k1=self.config.tokens_per_block)
            outs2 = torch.cat([outs2[:, :, :-1], outs1[:, :, -1:]], dim=2)

            outs2 = rearrange(outs2, 'b t k1 d -> b (t k1) d')
            outs.append(outs2)

            initial_state['state'] = [s_i[-1] for s_i in state_per_block]
            initial_state['n'] += tokens_emb_i.shape[1]
        return torch.cat(outs, dim=1)
            
    def compute_next_obs_pred_latents(self, incremental_state):
        assert incremental_state is not None and isinstance(incremental_state, dict)
        assert 'n' in incremental_state and 'state' in incremental_state
        assert len(incremental_state['state']) > 0 and incremental_state['state'][0] is not None
        batch_size = incremental_state['state'][0].shape[0]
        device = incremental_state['state'][0].device
        pred_tokens_emb = self._get_prediction_tokens_embeddings(batch_size, 1, device)[:, :-1]
    
        return self._model.forward_chunkwise(pred_tokens_emb, incremental_state['n'], incremental_state['state'])

    def _get_prediction_tokens_embeddings(self, batch_size: int, num_steps: int, device):
        if self.pred_tokens_version == 'shared':
            return self.placeholder_embeddings(torch.zeros(batch_size, num_steps * self.config.tokens_per_block, device=device).long())
        elif self.pred_tokens_version == 'per-token':
            tokens = torch.arange(self.config.tokens_per_block, device=device)
            tokens = rearrange(tokens, 'k1 -> 1 1 k1').expand(batch_size, num_steps, -1).flatten(1, 2)
            return self.placeholder_embeddings(tokens)
    
    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> tuple[LossWithIntermediateLosses, dict]:
        tokens = self._get_tokens(batch, tokenizer) if 'tokens' not in batch else batch['tokens']
        tokens = rearrange(tokens, 'b t k1 -> b (t k1)')

        if self.shared_embeddings:
            outputs = self(tokens, compute_next_state_latents=True, tokenizer=tokenizer)
        else:
            outputs = self(tokens, compute_next_state_latents=True)

        next_token_logits, next_token_labels = self.get_next_token_logits_and_labels(outputs, tokens, batch['mask_padding'])

        rewards_logits, labels_rewards, ends_logits, labels_ends = self.get_rewards_ends_logits_and_labels(
            outputs,
            batch['mask_padding'],
            batch['rewards'],
            batch['ends']
        )

        loss_obs = F.cross_entropy(next_token_logits, next_token_labels)
        loss_rewards = F.cross_entropy(rewards_logits, labels_rewards)
        loss_ends = F.cross_entropy(ends_logits, labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends), {}

    def get_next_token_logits_and_labels(self, outputs: torch.Tensor, tokens, mask_padding: torch.Tensor):
        pred_mask = mask_padding.clone()
        pred_mask[:, :self.context_length] = 0
        k1 = self.config.tokens_per_block
        pred_mask = pred_mask.unsqueeze(2).repeat(1, 1, k1)
        pred_mask[:, :, -1] = 0
        pred_mask = rearrange(pred_mask, 'b t k1 -> b (t k1)')

        labels = tokens.clone()
        labels = labels.masked_fill(torch.logical_not(pred_mask), -100)
        labels = labels.flatten()

        logits = self.head_observations(outputs.flatten(0, 1))

        return logits, labels

    def get_rewards_ends_logits_and_labels(self, outputs: torch.Tensor, mask_padding: torch.Tensor, rewards, ends):
        relevant_elements_mask = mask_padding.clone()
        relevant_elements_mask[:, :self.context_length] = 0

        action_tokens_logits = rearrange(outputs, 'b (t k1) e -> b t k1 e', k1=self.config.tokens_per_block)[:, :, -1]
        rewards_logits = self.head_rewards(action_tokens_logits.flatten(0, 1))
        ends_logits = self.head_ends(action_tokens_logits.flatten(0, 1))

        mask_fill = torch.logical_not(relevant_elements_mask)
        ignore_value = -100
        rewards_labels = (rewards.sign() + 1).masked_fill(mask_fill, ignore_value).long().reshape(-1)
        ends_labels = ends.masked_fill(mask_fill, ignore_value).reshape(-1)

        return rewards_logits, rewards_labels, ends_logits, ends_labels

    @property
    def uses_pop(self) -> bool:
        return True







