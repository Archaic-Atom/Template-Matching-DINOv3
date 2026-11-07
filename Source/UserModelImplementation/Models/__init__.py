# -*- coding: utf-8 -*
import JackFramework as jf

# TM_Model_A: baseline, now yaml-driven like the other variants — a thin
# ComponentBasedInterface subclass that only sets CONFIG_PATH. Its original
# _create_*.py / Loss/*.py / Nets/model.py were moved to _legacy/legacy_A/.
from .TM_Model_A import TMModelAInterface

# TM_Model_B/C/D/E: yaml-driven variants, each is a thin ComponentBasedInterface
# subclass that only sets CONFIG_PATH. Components live in _components/.
# E shares architecture with C and reuses C's trained ckpt — only differs at
# inference (soft sub-pixel vs hard argmax).
from .TM_Model_B import TMModelBInterface
from .TM_Model_C import TMModelCInterface
from .TM_Model_D import TMModelDInterface
from .TM_Model_E import TMModelEInterface

# TM_Model_F: learned-IP Stage-1 (W_s/W_d + VICReg rank floor) + frozen-SAR
# Stage-2 — the F0 "strong Stage-1, protected" counterfactual (2026-06-11).
from .TM_Model_F import TMModelFInterface

# TM_Model_G: TM_Model_F with freeze_sar_encoder flipped to false — the ONLY
# diff. F and G are the factorial control for the revised 2x2 (R2 comment 5a):
# one model family, one shared Stage-1 ckpt, freeze as a single-line toggle.
from .TM_Model_G import TMModelGInterface

# TM_Model_CB: TM_Model_C with architecture.matching.type flipped from
# CascadeHeadJBU to CascadeHeadBilinear — the ONLY diff. The guided-upsampling
# ablation control (R1 comment 7): same stride-16 1024-d feature source on both
# modalities, only the coarse ×4/×2 lift kernel changes.
from .TM_Model_CB import TMModelCBInterface

# TM_Model_CW: TM_Model_C with architecture.rgb_encoder.args.weights_path
# pointed at the web-pretrained LVD-1689M DINOv3 instead of the
# satellite-pretrained SAT-493M — the ONLY diff. The provider-corpus
# ablation control (R1 comment 5).
from .TM_Model_CW import TMModelCWInterface


def _get_model_dict() -> dict:
    return {
        'TM_Model_A': TMModelAInterface,
        'TM_Model_B': TMModelBInterface,
        'TM_Model_C': TMModelCInterface,
        'TM_Model_D': TMModelDInterface,
        'TM_Model_E': TMModelEInterface,
        'TM_Model_F': TMModelFInterface,
        'TM_Model_G': TMModelGInterface,
        'TM_Model_CB': TMModelCBInterface,
        'TM_Model_CW': TMModelCWInterface,
    }


def model_zoo(args: object, model_name: str) -> object:
    model_dict = _get_model_dict()
    assert model_name in model_dict, (
        f"Unknown --modelName {model_name!r}. Registered: {list(model_dict)}"
    )
    return model_dict[model_name](args)
