# -*- coding: utf-8 -*-
"""Shared building blocks for TM_Model variants.

Importing this package triggers @register decorators in each submodule,
populating COMPONENT_REGISTRY. The builder then looks up component types
by string name from YAML configs.
"""
from .registry import COMPONENT_REGISTRY, register, get_component, list_components

# Importing each submodule executes its @register calls.
# Modules are listed in dependency order (encoders before matching, losses last).
from . import sar_encoder              # noqa: F401  registers SarResNet16GN
from . import dino_backbone            # noqa: F401  registers DinoV3Backbone
from . import coarse_matching          # noqa: F401  registers CoarseMatchingHead
from . import fine_head_simple         # noqa: F401  registers FineFeatureHead
from . import fine_head_cascade        # noqa: F401  registers CascadeHead, CascadeHeadJBU
from . import sub_pixel                # noqa: F401  registers SubPixelDecoder
from . import alignment_loss           # noqa: F401  registers alignment_loss
from . import matching_loss            # noqa: F401  registers two_stage_matching_loss
from . import cascade_loss             # noqa: F401  registers cascade_loss

__all__ = ['COMPONENT_REGISTRY', 'register', 'get_component', 'list_components']
