# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kpool tail-cache attention backend for GLM-5.3-Flash (vendored).

vllm upstream commit 98ed0856f3 ("[Model] add GLM-5.3-Flash support" #53906)
ships ``KpoolTailBackend`` / ``KpoolTailMetadataBuilder`` /
``compute_kpool_tail_slot_mapping`` in
``vllm.v1.attention.backends.mla.indexer``. The vllm pin used by vllm-ascend
(``ba07e4a48f``, which predates #53906 and the KV-Cache Layout Refactor #51718)
does not contain them, so vllm-ascend vendors the three pieces here.

This is a near-verbatim copy of the upstream implementation. The only
intentional divergence is dropping ``supported_kv_cache_layouts`` (an abstract
method that #51718 added to the base ``AttentionBackend``) because this vllm
predates that refactor -- the method does not exist on the base class here, so
overriding it would be dead code.

The tail cache is a storage-only circular buffer (one ``index_kpool``-slot
block per request, overwritten in place by ``pos % kpool``); the backend's sole
job is to publish the circular slot mapping via its metadata builder. The pool
*content* (softmax-weighted sum vs keep-every-Nth) is computed by the kpool
compress kernel inside the indexer op.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec


def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map every token to its request's one circular tail block.

    Each request owns a single tail block of ``kpool`` slots; token ``p`` of
    that request is mapped to ``own_block * kpool + p % kpool`` so the in-progress
    pool's raw K + gate score is overwritten in place as decode advances.
    """
    if out is None:
        out = slot_mapping.clone()
    else:
        assert out.shape == slot_mapping.shape
        out.copy_(slot_mapping)
    if num_actual_tokens == 0:
        return out
    tokens = torch.arange(num_actual_tokens, device=slot_mapping.device)
    req = torch.searchsorted(query_start_loc, tokens, right=True) - 1
    req = req.clamp_(min=0, max=num_reqs - 1)
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    return out


class KpoolTailMetadataBuilder(AttentionMetadataBuilder):
    """Build only the circular slot mapping needed by the storage-only tail."""

    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.slot_mapping_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata)
        )
        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            slot_mapping_buffer = self.slot_mapping_buffer[
                : slot_mapping.numel()
            ].view_as(slot_mapping)
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
                out=slot_mapping_buffer,
            )
        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )


class KpoolTailBackend(DeepseekV32IndexerBackend):
    """Storage-only backend for the GLM-5.3-Flash kpool tail cache.

    Vendored from vllm 98ed0856f3 (#53906); ``supported_kv_cache_layouts``
    (added by KV-Cache Layout Refactor #51718) is omitted because the vllm pin
    predates that refactor and the base ``AttentionBackend`` here has no such
    method. ``get_kv_cache_shape`` / ``get_kv_cache_stride_order`` are
    overridden because the base ``DeepseekV32IndexerBackend`` asserts
    ``num_kv_heads == 1`` and returns a 3D layout, while ``KpoolTailSpec``
    uses ``num_kv_heads == 2`` (K + gate score) and needs the 4D layout the
    kpool reshape branch in ``model_runner_v1._reshape_kv_cache_tensors``
    views the raw tensor as.
    """

    @staticmethod
    def get_name() -> str:
        return "KPOOL_TAIL"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type[KpoolTailMetadataBuilder]:
        return KpoolTailMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # KpoolTailSpec uses num_kv_heads=2 (K + gate score); the base
        # DeepseekV32IndexerBackend asserts num_kv_heads==1 (single MLA latent)
        # and returns a 3D (num_blocks, block_size, head_size) layout. Publish
        # the 4D (num_blocks, block_size, num_kv_heads, head_size) layout so
        # raw_typed.view(kv_cache_shape) matches the tail's [block, head,
        # head_dim] content.
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # 4D content layout (see get_kv_cache_shape); identity permutation.
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)
