# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""Check action normalization through fresh and checkpoint-backed pipelines."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

pytest.importorskip("transformers")
pytest.importorskip("scipy")

processor = pytest.importorskip("lerobot.policies.molmoact2.processor_molmoact2")


IMAGE = f"{OBS_IMAGES}.image"


def _stats(center: float) -> dict[str, dict[str, torch.Tensor]]:
    return {
        ACTION: {
            "mean": torch.tensor([center, 0.0]),
            "std": torch.tensor([10.0, 1.0]),
            "q01": torch.tensor([center - 20.0, -0.5]),
            "q99": torch.tensor([center + 20.0, 0.5]),
            "q10": torch.tensor([center - 15.0, -0.5]),
            "q90": torch.tensor([center + 15.0, 0.5]),
            "min": torch.tensor([center - 30.0, -0.5]),
            "max": torch.tensor([center + 30.0, 0.5]),
            "mask": torch.tensor([True, False]),
        },
    }


class _PromptProcessor:
    tokenizer = SimpleNamespace(eos_token="", eos_token_id=2)

    def __call__(self, *, text: list[str], **kwargs) -> dict[str, torch.Tensor]:
        del kwargs
        ids = torch.tensor([[1, 3]]).repeat(len(text), 1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.mark.parametrize("route", ["fresh", "legacy_reload", "fine_tune"])
@pytest.mark.parametrize(
    ("action_mode", "norm_mode", "scale"),
    [
        ("continuous", NormalizationMode.MEAN_STD, 10.0),
        ("continuous", NormalizationMode.QUANTILES, 20.0),
        ("continuous", NormalizationMode.QUANTILE10, 15.0),
        ("continuous", NormalizationMode.MIN_MAX, 30.0),
        ("discrete", NormalizationMode.MEAN_STD, 10.0),
        ("both", NormalizationMode.MEAN_STD, 10.0),
    ],
)
@pytest.mark.parametrize("stage", ["pre", "post"])
def test_action_scaling_follows_effective_normalizer(
    tmp_path, monkeypatch, route, action_mode, norm_mode, scale, stage
):
    monkeypatch.setattr(processor, "_load_local_molmoact2_processor", lambda _: _PromptProcessor())
    monkeypatch.setattr(
        processor,
        "_single_token_id",
        lambda tokenizer, token: 1 if token == processor.ACTION_START_TOKEN else 3,
    )
    monkeypatch.setattr(
        processor, "UniversalActionProcessor", SimpleNamespace(from_pretrained_local=lambda _: object())
    )
    fast_inputs = []

    def capture_fast_input(action: np.ndarray, action_processor: object) -> str:
        del action_processor
        fast_inputs.append(action.copy())
        return f"{processor.ACTION_START_TOKEN}{processor.ACTION_END_TOKEN}"

    monkeypatch.setattr(processor, "_build_discrete_action_string", capture_fast_input)
    config = MolmoAct2Config(
        checkpoint_path=str(tmp_path),
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        action_mode=action_mode,
        inference_action_mode="discrete" if action_mode == "discrete" else "continuous",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        normalization_mapping={"ACTION": norm_mode, "STATE": NormalizationMode.IDENTITY},
    )
    active_map = dict(config.normalization_mapping)
    if route == "fine_tune":
        config.normalization_mapping["ACTION"] = NormalizationMode.QUANTILES
    pre, post = make_pre_post_processors(config, dataset_stats=_stats(0.0 if route == "fine_tune" else 20.0))
    if route != "fresh":
        for pipeline in (pre, post):
            pipeline.save_pretrained(tmp_path)
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            path = tmp_path / name
            data = json.loads(path.read_text(encoding="utf-8"))
            for step in data["steps"]:
                if step["registry_name"] in {"molmoact2_clamp_action", "molmoact2_clamp_normalized"}:
                    step["config"].pop("clamp_action", None)
                    step["config"].pop("normalization_mask", None)
            path.write_text(json.dumps(data), encoding="utf-8")
        if route == "legacy_reload":
            config.normalization_mapping["ACTION"] = NormalizationMode.IDENTITY
            pre, post = make_pre_post_processors(config, pretrained_path=str(tmp_path))
        else:
            config.normalization_mapping = active_map
            override = {"stats": _stats(20.0), "norm_map": active_map}
            pre, post = make_pre_post_processors(
                config,
                pretrained_path=str(tmp_path),
                preprocessor_overrides={"normalizer_processor": override},
                postprocessor_overrides={"unnormalizer_processor": override},
            )

    is_unbounded = action_mode == "continuous" and norm_mode == NormalizationMode.MEAN_STD
    if stage == "pre":
        action = torch.tensor([-10.0, 0.0, 20.0, 50.0])
        gripper = torch.tensor([0.5, -0.25, 0.0, 0.1])
        batch = {
            OBS_STATE: torch.zeros(1, 2),
            IMAGE: torch.zeros(1, 3, 4, 4),
            ACTION: torch.stack((action, gripper), dim=-1).unsqueeze(0),
            "task": ["push the block"],
        }
        expected_arm = (action - 20.0) / scale
        if not is_unbounded:
            expected_arm = expected_arm.clamp(-1.0, 1.0)
        expected = torch.stack((expected_arm, gripper), dim=-1)
        outputs = pre(batch)
        torch.testing.assert_close(outputs[ACTION][0, :, :2], expected, atol=3e-6, rtol=0)
        if action_mode != "continuous":
            assert len(fast_inputs) == 1
            np.testing.assert_allclose(fast_inputs[0], expected.numpy(), atol=3e-6, rtol=0)
    else:
        prediction = torch.tensor([-2.5, -1.25, 0.25, 2.75])
        gripper = torch.tensor([1.25, -1.5, 0.0, 0.1])
        effective = prediction if is_unbounded else prediction.clamp(-1.0, 1.0)
        outputs = post(torch.stack((prediction, gripper), dim=-1))
        torch.testing.assert_close(outputs[..., 0], 20.0 + scale * effective, atol=4e-6, rtol=0)
        torch.testing.assert_close(outputs[..., 1], gripper.clamp(-1.0, 1.0), atol=2e-6, rtol=0)
