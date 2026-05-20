# -*- coding: utf-8 -*-
"""DINO backbone wrapper.

Wraps torch.hub-loaded DINO ViTs so that `forward(x)` returns the list of
intermediate-layer features needed by downstream matching heads. The DINO
model is frozen at construction (configure freeze via interface, not here).
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

from .registry import register


_LAYER_IDS = {
    'dinov3_vith16plus': [7, 15, 23, 31],
    'dinov3_vitl16':     [4, 11, 17, 23],
    'dinov2_vitl14':     [4, 11, 17, 23],
}


def _get_layer_ids(model_name: str) -> list[int]:
    if model_name not in _LAYER_IDS:
        raise KeyError(f"Unknown DINO model '{model_name}'. Known: {list(_LAYER_IDS)}")
    return _LAYER_IDS[model_name]


@register("DinoV3Backbone")
class DinoV3Backbone(nn.Module):
    """Loads a DINOv3 ViT via torch.hub and exposes intermediate-layer features.

    Args:
        name: ViT variant, e.g. 'dinov3_vitl16'.
        repo_path: filesystem path to the dinov3 hub repo (passed to torch.hub).
        weights_path: filesystem path to the pretrained .pth file.
        layer_ids: optional override of which intermediate layers to return.

    Freezing is configured at the variant interface layer (config.architecture.
    rgb_encoder.freeze) — not here — so all variants share the same constructor.
    """

    def __init__(
        self,
        name: str = 'dinov3_vitl16',
        repo_path: str = '',
        weights_path: str = '',
        layer_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"DINO repo path not found: {repo_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"DINO weights not found: {weights_path}")

        self.name = name
        self.dino = torch.hub.load(repo_path, name, source='local', weights=weights_path)
        self.layer_ids = layer_ids if layer_ids is not None else _get_layer_ids(name)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns intermediate-layer features at stride 16 (reshape=True)."""
        return self.dino.get_intermediate_layers(
            x, n=self.layer_ids,
            reshape=True, return_class_token=False, norm=False,
        )
