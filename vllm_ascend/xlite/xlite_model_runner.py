#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#
"""xlite model runner subclasses (MRV1/MRV2) for `vllm-ascend`"""

from collections.abc import Iterable
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2
from vllm_ascend.xlite.utils import _TensorRepr
from vllm_ascend.xlite.xlite import XliteWrapper


class XliteModelRunnerMixin:
    fallback_model: nn.Module
    """The fallback model from the native :class:`NPUModelRunner` implementation."""
    runner_cls: type[XliteWrapper] = XliteWrapper
    """The class of the xlite model forward backend; consistent with :data:`runner_model`."""
    runner_model: XliteWrapper
    """The xlite model forward backend; consistent with :data:`runner_cls`."""
    _runner_enabled: bool = False
    """If :data:`runner_model` is the current model forward backend; otherwise, :data:`fallback_model`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert get_ascend_config().xlite_graph_config.enabled, "xlite graph must be enabled to use XliteModelRunner"
        super().__init__(*args, **kwargs)

    @contextmanager
    def _bypass_xlite_wrapper(self):
        """Temporarily route ``self.model`` to the native runnable.

        Profile runs and dummy runs (including ``execute_dummy_batch`` and cudagraph capture warmups) should not enter
        the xlite forward path.
        """
        if not self.runner_enabled:
            yield
            return

        self.model = self.fallback_model
        try:
            yield
        finally:
            self.model = self.runner_model  # type: ignore[assignment]

    @property
    def model(self) -> nn.Module:
        """The current model forward backend, either :data:`runner_model` or :data:`fallback_model`."""
        return self._model

    @model.setter
    def model(self, value: nn.Module) -> None:
        self._model = value
        self._runner_enabled = isinstance(value, self.runner_cls)

    @model.deleter
    def model(self) -> None:
        self._runner_enabled = False
        del self._model
        del self.fallback_model
        del self.runner_model

    @property
    def runner_enabled(self) -> bool:
        """If the current model forward backend is the xlite runner."""
        return self._runner_enabled

    def _get_flattened_kv_caches(
        self, kv_caches: list[torch.Tensor] | list[Iterable[torch.Tensor]], /, *, check_is_contiguous: bool = True
    ) -> list[torch.Tensor]:
        """Flattens the KV cache list to a single list of tensors.

        Args:
            kv_caches (list[torch.Tensor] | list[Iterable[torch.Tensor]]): The KV cache list to flatten.
            check_is_contiguous (bool, optional): If True, checks that each retrieved tensor is contiguous in memory. If
                any tensor is not contiguous, a ValueError is raised. Default is True.

        Returns:
            list[torch.Tensor]: The flattened KV cache list.
        """
        flattened_kv_caches: list[torch.Tensor] = []
        for cache in kv_caches:
            if isinstance(cache, torch.Tensor):
                flattened_kv_caches.append(cache)
            elif isinstance(cache, Iterable) and all(isinstance(c, torch.Tensor) for c in cache):  # type: ignore[arg-type]
                flattened_kv_caches.extend(cache)
            else:
                raise TypeError(f"Unexpected kv_cache type: {type(cache)}")
        if check_is_contiguous and not all(tensor.is_contiguous() for tensor in flattened_kv_caches):
            raise ValueError("xlite: Found non-contiguous tensor in the KV caches.")
        return flattened_kv_caches


class XliteModelRunner(XliteModelRunnerMixin, NPUModelRunner):  # type: ignore[misc]
    """xlite model runner for model runner V1 (`vllm_ascend/worker/model_runner_v1.py`).

    Remove this class together with `NPUModelRunner` V1 when the V1 runner is dropped; the mixin and
    `XliteModelRunnerV2` below stay.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
            logger.error("xlite does not currently support Sparse SFA/LI C8.")

    def get_model(self) -> nn.Module:
        """Returns the unwrapper fallback model. See :meth:`NPUModelRunner.get_model` for details."""
        with self._bypass_xlite_wrapper():
            return super().get_model()

    def load_model(self) -> None:
        super().load_model()
        self.fallback_model = self.model
        # NOTE: this will create a circular reference between XliteModelRunner and XliteWrapper instances,
        # but this should be fine since they are both long-lived objects
        self.model = self.runner_model = self.runner_cls(self, self.vllm_config, device=self.device)  # type: ignore[assignment]
        # EAGLE3: super().load_model() configured aux hidden-state layers on the fallback
        # model (it ran while self.model was still the fallback, before the wrapper was
        # installed). Re-apply the same resolution to the wrapper so __call__ returns
        # (h, [aux...]); the wrapper stores the tuple (sizing __call__'s aux buffers) and
        # re-syncs the fallback for its overflow path. The xlite C++ model's
        # aux_layer_ids is baked into xlite_config before init inside the wrapper ctor.
        if getattr(self, "use_aux_hidden_state_outputs", False):
            aux_layers = self._get_eagle3_aux_layers_from_config()
            if not aux_layers:
                aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()
            self.model.set_aux_hidden_state_layers(aux_layers)

    def _dummy_run(
        self,
        *args: Any,
        is_profile: bool = False,
        is_graph_capturing: bool = False,
        with_prefill: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route dummy/profile runs to the native runnable, bypassing the xlite wrapper.

        See :meth:`_bypass_xlite_wrapper` for why dummy runs must not trigger the xlite forward. Delegates every
        argument to the base implementation unchanged.

        Exception: in ``full_mode`` full-takeover (xlite preallocates the prefill-sized token budget, so serving
        never falls back to the native runnable for batches within the scheduler cap) and no speculative
        decoding is configured, the main profile run is a no-op rather than a native dummy forward. Serving
        prefill intermediates live in the xlite C++ tensor pool (allocated via ``aclrtMalloc`` and already
        counted under ``non_torch``), so the native dummy run would measure a fallback activation peak that is
        never reached at serving time — a pure virtual reservation that shrinks the KV cache. Skipping it keeps
        ``torch_peak_increase`` at the load_model baseline (verified: serving prefill adds no torch-side peak),
        returning that HBM to the KV cache.

        Three cases keep the native dummy forward even under full_mode (i.e. the no-op does not apply):
        - ``with_prefill=True``: the MC2 capacity pre-run (``model_runner_v1.py`` profile_run) reserves the
          native MC2/HCCL communication buffers used by the eager overflow path.
        - ``is_graph_capturing``: full_mode forces ``enforce_eager``, so capture warmups keep the native path
          consistent with the eager fallback used on overflow.
        - ``speculative_config`` set (MTP/EAGLE3/DSpark/extract_hidden_states/...): the drafter runs on the
          native runnable and EAGLE3's aux-capture path routes through ``npu_runnable`` when ``inputs_embeds``
          is set (xlite.py), so the fallback activation peak must be profiled to reserve draft/aux memory.

        The no-op returns the xlite wrapper's preallocated ``hidden_states`` (sized to ``max_m`` at load time)
        so downstream ``dummy_compute_logits`` / sampler dummy runs see a correctly-shaped tensor without
        allocating anything new.
        """
        if not is_profile and not is_graph_capturing:
            # DP `excute_dummy_batch` must be routed to xlite forward path to avoid out of sync issues
            return super()._dummy_run(*args, is_profile=is_profile, is_graph_capturing=is_graph_capturing, with_prefill=with_prefill, **kwargs)  # type: ignore[return-value]

        if is_profile and not with_prefill and not is_graph_capturing and self.runner_enabled:
            xlite_cfg = get_ascend_config().xlite_graph_config
            # Full-takeover guard: only when the xlite token budget equals the scheduler cap, i.e. every serving
            # batch fits within `max_tokens` and never overflows to the native runnable. decode_only sizes the
            # budget to a decode step, so prefill still runs on the native path and must profile there.
            # Speculative decoding (MTP/EAGLE3/DSpark/extract_hidden_states/...) is also excluded: the drafter
            # runs on the native runnable (its `dummy_run` is inside `super()._dummy_run`), and EAGLE3's aux
            # path routes through `npu_runnable` when `inputs_embeds` is set (xlite.py), so serving still
            # reaches the fallback and its activation peak must be profiled. The no-op would skip
            # `drafter.dummy_run` and under-reserve draft/aux memory, risking serving-time OOM.
            full_takeover = (
                xlite_cfg.full_mode
                and not self.speculative_config
                and self.runner_model.max_tokens
                >= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            if full_takeover:
                torch.npu.reset_peak_memory_stats(self.device)
                before = torch.npu.memory_allocated(self.device)
                # Return the preallocated hidden_states buffer (zero-cost: allocated at load_model, already
                # counted in the load baseline). profile_run expects (hidden_states, last_hidden_states); use
                # the same buffer for both since downstream only inspects shape/dtype for dummy sampler/logits.
                h = self.runner_model.hidden_states
                logger.info_once(
                    "xlite full_takeover: profile run skipped native dummy forward; torch peak %.2f MiB "
                    "(before %.2f MiB, delta %.2f MiB) — fallback activation peak no longer reserved.",
                    torch.npu.max_memory_allocated(self.device) / (1 << 20),
                    before / (1 << 20),
                    (torch.npu.max_memory_allocated(self.device) - before) / (1 << 20),
                )
                return h, h

        with self._bypass_xlite_wrapper():
            return super()._dummy_run(*args, is_profile=is_profile, is_graph_capturing=is_graph_capturing, with_prefill=with_prefill, **kwargs)  # type: ignore[return-value]

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, kv_cache_allocation_context: AbstractContextManager | None = None
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context=kv_cache_allocation_context)
        kv_caches: list[torch.Tensor] = self._get_flattened_kv_caches(self.kv_caches)
        logger.info_once("xlite: registering kv caches to runner model: %s", _TensorRepr(kv_caches, max_it=None))
        self.runner_model.register_kv_caches(kv_caches)

        # Check attention metadata backend compatibility. Probe the first standard attention group
        # (first kv-cache group's first attention group) — the same layer the xlite wrapper forwards
        # (``model.layers.0.self_attn.attn``). ``[-1]`` would instead pick the last group, which for
        # speculative methods like extract_hidden_states/dspark is the ``CacheOnlyAttentionLayer``
        # group and triggers a false mismatch against the xlite-expected ``AscendMetadata``.
        ascend_metadata_builder = self.attn_groups[0][0].get_metadata_builder(0)
        ascend_metadata_cls = getattr(ascend_metadata_builder, "metadata_cls", ascend_metadata_builder)
        xlite_expects = self.runner_model.adapter_xlite_model._attn_metadata_type
        if ascend_metadata_cls != xlite_expects and (
            not isinstance(xlite_expects, tuple) or ascend_metadata_cls not in xlite_expects
        ):
            logger.error(
                "Attention metadata mismatch: xlite expects (one of) %s, but got %s. Be aware of runtime issues.",
                xlite_expects,
                ascend_metadata_cls,
            )

    def _should_build_dummy_attn_metadata(
        self, force_attention: bool = False, is_profile: bool = False, *args: Any, **kwargs: Any
    ) -> bool:
        base_condition = super()._should_build_dummy_attn_metadata(force_attention, is_profile, *args, **kwargs)
        return base_condition or (self.runner_enabled and not is_profile)


class XliteModelRunnerV2(XliteModelRunnerMixin, NPUModelRunnerV2):  # type: ignore[misc]
    """xlite model runner for model runner V2 (`vllm_ascend/worker/v2/model_runner.py`).

    Keeps all V2-only adaptations (KV cache regrouping, `decode_query_len` alias, sparse C8 guard,
    `load_model` signature) here so the V1 class above can be removed without touching it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        enable_sparse_sfa_c8 = self.vllm_config.cache_config.cache_dtype in ["fp8", "int8"]
        enable_sparse_li_c8 = self.vllm_config.attention_config.indexer_kv_dtype in ["fp8", "int8"]
        if enable_sparse_sfa_c8 or enable_sparse_li_c8:
            logger.error("xlite does not currently support Sparse SFA/LI C8.")

    @property
    def uniform_decode_query_len(self) -> int:
        """V1-compatibility alias for the V2 ``decode_query_len`` attribute.

        ``NPUWorker.execute_dummy_batch`` reads the V1 attribute name via ``getattr`` to size DP dummy
        batches; the attribute was renamed in V2, and without this alias the fallback would silently
        size them to a single token per rank.
        """
        return self.decode_query_len

    def get_model(self) -> nn.Module:
        """Returns the unwrapper fallback model. See :meth:`NPUModelRunner.get_model` for details."""
        with self._bypass_xlite_wrapper():
            return super().get_model()

    def load_model(self, load_dummy_weights: bool = False, *args: Any, **kwargs: Any) -> None:
        super().load_model(load_dummy_weights, *args, **kwargs)
        self.fallback_model = self.model
        # NOTE: this will create a circular reference between XliteModelRunner and XliteWrapper instances,
        # but this should be fine since they are both long-lived objects
        self.model = self.runner_model = self.runner_cls(self, self.vllm_config, device=self.device)  # type: ignore[assignment]
        # EAGLE3: super().load_model() configured aux hidden-state layers on the fallback
        # model (it ran while self.model was still the fallback, before the wrapper was
        # installed). Re-apply the same resolution to the wrapper so __call__ returns
        # (h, [aux...]); the wrapper stores the tuple (sizing __call__'s aux buffers) and
        # re-syncs the fallback for its overflow path. The xlite C++ model's
        # aux_layer_ids is baked into xlite_config before init inside the wrapper ctor.
        if getattr(self, "use_aux_hidden_state_outputs", False):
            aux_layers = self._get_eagle3_aux_layers_from_config()
            if not aux_layers:
                aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()
            self.model.set_aux_hidden_state_layers(aux_layers)

    def _dummy_run(self, *args: Any, is_profile: bool = False, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Route dummy/profile runs to the native runnable, bypassing the xlite wrapper.

        See :meth:`_bypass_xlite_wrapper` for why dummy runs must not trigger the xlite forward. Delegates every
        argument to the base implementation unchanged.
        """
        if not is_profile:
            # DP `excute_dummy_batch` must be routed to xlite forward path to avoid out of sync issues
            return super()._dummy_run(*args, is_profile=is_profile, **kwargs)  # type: ignore[return-value]

        with self._bypass_xlite_wrapper():
            return super()._dummy_run(*args, is_profile=is_profile, **kwargs)  # type: ignore[return-value]

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, kv_cache_allocation_context: AbstractContextManager | None = None
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context=kv_cache_allocation_context)
        kv_caches: list[torch.Tensor] = self._get_flattened_kv_caches(self.kv_caches, check_is_contiguous=False)
        logger.info_once("xlite: registering kv caches to runner model: %s", _TensorRepr(kv_caches, max_it=None))
        self.runner_model.register_kv_caches(kv_caches)

        # Check attention metadata backend compatibility. Probe the first standard attention group
        # (first kv-cache group's first attention group) — the same layer the xlite wrapper forwards
        # (``model.layers.0.self_attn.attn``). ``[-1]`` would instead pick the last group, which for
        # speculative methods like extract_hidden_states/dspark is the ``CacheOnlyAttentionLayer``
        # group and triggers a false mismatch against the xlite-expected ``AscendMetadata``.
        ascend_metadata_builder = self.attn_groups[0][0].get_metadata_builder(0)
        ascend_metadata_cls = getattr(ascend_metadata_builder, "metadata_cls", ascend_metadata_builder)
        xlite_expects = self.runner_model.adapter_xlite_model._attn_metadata_type
        if ascend_metadata_cls != xlite_expects and (
            not isinstance(xlite_expects, tuple) or ascend_metadata_cls not in xlite_expects
        ):
            logger.error(
                "Attention metadata mismatch: xlite expects (one of) %s, but got %s. Be aware of runtime issues.",
                xlite_expects,
                ascend_metadata_cls,
            )
