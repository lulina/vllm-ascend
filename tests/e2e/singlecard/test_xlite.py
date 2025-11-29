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
#
"""
Compare the outputs of vLLM with and without xlite.

Run `pytest tests/e2e/singlecard/test_xlite.py`.
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_with_xlite_decode_only(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_XLITE": "1"}):
        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=False,
        ) as runner:
            vllm_xlite_outputs = runner.model.generate(prompts,
                                                       sampling_params)

    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=True,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)
    vllm_xlite_outputs_list = []
    for output in vllm_xlite_outputs:
        vllm_xlite_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_xlite_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_xlite_outputs",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_models_with_xlite_full_mode(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_XLITE": "2"}):
        with VllmRunner(
                model,
                max_model_len=1024,
                enforce_eager=False,
        ) as runner:
            vllm_xlite_outputs = runner.model.generate(prompts,
                                                       sampling_params)

    with VllmRunner(
            model,
            max_model_len=1024,
            enforce_eager=True,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)
    vllm_xlite_outputs_list = []
    for output in vllm_xlite_outputs:
        vllm_xlite_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_xlite_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_xlite_outputs",
    )