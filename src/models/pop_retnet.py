# Based on https://github.com/fkodom/yet-another-retnet/blob/main/yet_another_retnet/retention.py

from typing import Union, Callable, Optional, List, Sequence, Tuple
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import yet_another_retnet as yar
import yet_another_retnet.retention
from einops import rearrange, einsum
from torch import Tensor
from yet_another_retnet.retention import ActivationString, _build_decay_gammas, _build_decay_mask
from yet_another_retnet.retnet import RetNetDecoder


def retention_chunkwise_per_block_states(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    tokens_per_block: int,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    decay_gammas = _build_decay_gammas(
        num_heads=query.shape[1], device=query.device, dtype=query.dtype
    )
    decay_mask = _build_decay_mask(
        num_heads=query.shape[1],
        query_length=query.shape[2],
        key_length=key.shape[2],
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: head_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    # intra-chunk (same as parallel retention)
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(decay_gammas, "h -> () h () ()")
    inner_pos = rearrange(
        torch.arange(key.size(2), device=key.device, dtype=key.dtype) + 1,
        "n -> () () n ()",
    )
    per_frame_inner_pos = rearrange(
        torch.arange(tokens_per_block, device=key.device, dtype=key.dtype) + 1,
        "k1 -> () () k1 ()",
    )

    per_frame_state_decays = decay_gammas ** (tokens_per_block - per_frame_inner_pos)

    key = rearrange(key, 'b h (t k1) d -> b h t k1 d', k1=tokens_per_block)
    value = rearrange(value, 'b h (t k1) d -> b h t k1 d', k1=tokens_per_block)
    discounted_key = einsum(key, per_frame_state_decays, 'b h t k1 d, _ h k1 _ -> b h t k1 d')
    state = einsum(discounted_key, value, 'b h t k1 d1, b h t k1 d2 -> b h t d1 d2')

    per_frame_decay_gammas = decay_gammas ** tokens_per_block
    if prev_state is not None:
        state[:, :, 0] = state[:, :, 0] + per_frame_decay_gammas * prev_state
    for i in range(1, state.shape[2]):
        state[:, :, i] = state[:, :, i] + per_frame_decay_gammas * state[:, :, i-1]  # b h d1 d2

    # For ease of using these states for our purposes, rearrange so that it is in the batch dim:
    state = rearrange(state, 'b h t d1 d2 -> t b h d1 d2')

    if prev_state is not None:
        # Update the retention Tensor, based on cross-chunk information
        inner_decay = decay_gammas ** inner_pos
        retention = retention + (
                einsum(query, prev_state, "b h n d1, b h d1 d2 -> b h n d2") * inner_decay
        )

    return retention, state


def _multiply_by_i(x: Tensor) -> Tensor:
    """Multiply a complex-valued tensor by the imaginary unit 'i'."""
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)


def _theta_shift(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # TODO: Add docstring
    return (x * cos) + (_multiply_by_i(x) * sin)


def retention_chunkwise(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    scale: Optional[float] = None,
    compute_state: bool = True
) -> Tuple[Tensor, Tensor]:
    decay_gammas = _build_decay_gammas(
        num_heads=query.shape[1], device=query.device, dtype=query.dtype
    )
    decay_mask = _build_decay_mask(
        num_heads=query.shape[1],
        query_length=query.shape[2],
        key_length=key.shape[2],
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: head_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    # intra-chunk (same as parallel retention)
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(decay_gammas, "h -> () h () ()")
    inner_pos = rearrange(
        torch.arange(key.size(2), device=key.device, dtype=key.dtype) + 1,
        "n -> () () n ()",
    )
    if compute_state:
        state_decays = decay_gammas ** (key.size(2) - inner_pos)
        discounted_key = einsum(key, state_decays, 'b h n d, _ h n _ -> b h n d')
        state = einsum(discounted_key, value, 'b h n d1, b h n d2 -> b h d1 d2')

        if prev_state is not None:
            # Update internal state to return to the user
            chunk_decay = decay_gammas ** key.size(2)
            state = state + prev_state * chunk_decay
    else:
        state = prev_state

    if prev_state is not None:
        # Update the retention Tensor, based on cross-chunk information
        inner_decay = decay_gammas**inner_pos
        retention = retention + (
            einsum(query, prev_state, "b h n d1, b h d1 d2 -> b h n d2") * inner_decay
        )

    return retention, state


class POPMultiScaleRetention(yet_another_retnet.retention.MultiScaleRetention):
    def forward_chunkwise_per_block_state(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        start_idx: int,
        tokens_per_block: int,
        prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            # Combined (cross + intra chunk):
            indices = torch.arange(start_idx, start_idx + q.size(2), device=q.device, dtype=q.dtype)
            indices = rearrange(indices, "n -> () () n ()")
            thetas = rearrange(self.thetas, "d -> () () () d")
            angles = indices * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)
            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, state = retention_chunkwise_per_block_states(q, k, v, prev_state=prev_state, tokens_per_block=tokens_per_block)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward_chunkwise_from_per_block_states(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        start_idx: int,
        tokens_per_block: int,
        prev_states_per_block: Optional[Tensor],
        compute_state: bool = True
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            # Combined (cross + intra chunk):
            indices = torch.arange(start_idx, start_idx + q.size(2), device=q.device, dtype=q.dtype)
            indices = rearrange(indices, "n -> () () n ()")
            thetas = rearrange(self.thetas, "d -> () () () d")
            angles = indices * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)
            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # convert to a batch of time steps:
        q = rearrange(q, 'b h (t k1) d -> (t b) h k1 d', k1=tokens_per_block)
        k = rearrange(k, 'b h (t k1) d -> (t b) h k1 d', k1=tokens_per_block)
        v = rearrange(v, 'b h (t k1) d -> (t b) h k1 d', k1=tokens_per_block)

        # shift the states:
        t = prev_states_per_block.shape[0]
        prev_states_per_block = rearrange(prev_states_per_block, 't b h d1 d2 -> (t b) h d1 d2')

        # Apply retention then group norm.
        retention, state = retention_chunkwise(q, k, v, prev_state=prev_states_per_block, compute_state=compute_state)
        retention = rearrange(retention, '(t b) h k1 d -> b h (t k1) d', k1=tokens_per_block, t=t)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, state


class POPRetNetDecoderLayer(yet_another_retnet.retnet.RetNetDecoderLayer):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish", norm_first: bool = True,
                 layer_norm_eps: float = 1e-6, device: Optional[Union[torch.device, str]] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_first, layer_norm_eps, device,
                         dtype)
        self.retention = POPMultiScaleRetention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )

    def forward_chunkwise_per_block_state(
        self, x: Tensor, start_idx: int, tokens_per_block: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_chunkwise_per_block_state(
                x, x, x, start_idx=start_idx, tokens_per_block=tokens_per_block, prev_state=prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state

    def forward_chunkwise_from_per_block_states(
        self, x: Tensor, start_idx: int, tokens_per_block: int, prev_states_per_block: Optional[Tensor] = None, compute_state: bool = True
    ) -> Tuple[Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_chunkwise_from_per_block_states(
                x, x, x, start_idx=start_idx, tokens_per_block=tokens_per_block, prev_states_per_block=prev_states_per_block, compute_state=compute_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state


class POPRetNetDecoder(yet_another_retnet.retnet.RetNetDecoder):

    def forward_chunkwise_per_block_state(
            self, x: Tensor, start_idx: int, tokens_per_block: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        if not prev_states:
            prev_states = [None] * self.num_layers
        elif len(prev_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(prev_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state in zip(self.layers, prev_states):
            assert isinstance(layer, POPRetNetDecoderLayer)
            x, state = layer.forward_chunkwise_per_block_state(x, start_idx, tokens_per_block, prev_state)
            states.append(state)
        return x, states

    def forward_chunkwise_from_per_block_states(
            self, x: Tensor, start_idx: int, tokens_per_block: int, per_block_states: Sequence[Optional[Tensor]] = (), compute_state: bool = True
    ) -> Tuple[Tensor, List[Tensor]]:
        if not per_block_states:
            per_block_states = [None] * self.num_layers
        elif len(per_block_states) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} previous states, got {len(per_block_states)}"
            )

        states: List[Tensor] = []
        for layer, prev_state_pf in zip(self.layers, per_block_states):
            assert isinstance(layer, POPRetNetDecoderLayer)
            x, state = layer.forward_chunkwise_from_per_block_states(x, start_idx, tokens_per_block, prev_state_pf, compute_state=compute_state)
            states.append(state)
        return x, states
    

def test_per_frame_chunkwise_states():
    DEVICE = "cuda:0"
    DTYPE = torch.float32
    batch_size, num_heads, seq_length, hidden_dim = 5, 4, 10*65, 256

    size = (batch_size, num_heads, seq_length, hidden_dim)
    query = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    key = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    value = torch.randn(*size, device=DEVICE, dtype=DTYPE)

    r1, s1 = retention_chunkwise_per_block_states(query, key, value, prev_state=None, tokens_per_block=64)

    r2, s2 = yar.retention.retention_chunkwise(query, key, value, prev_state=None)

    torch.testing.assert_close(s1[-1], s2)
    torch.testing.assert_close(r1, r2)


if __name__ == '__main__':
    test_per_frame_chunkwise_states()

