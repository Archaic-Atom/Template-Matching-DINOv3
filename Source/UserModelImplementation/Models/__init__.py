# -*- coding: utf-8 -*
import JackFramework as jf

# TM_Model_A: legacy path (its own _create_*.py + Loss/*.py + Nets/model.py).
# A is currently training under nohup — its files are intentionally NOT
# touched. Phase 2 will refactor A onto the yaml-driven path after Stage 1
# finishes and we've verified state_dict compatibility.
from .TM_Model_A import TMModelAInterface

# TM_Model_B/C/D: yaml-driven variants, each is a thin ComponentBasedInterface
# subclass that only sets CONFIG_PATH. Components live in _components/.
from .TM_Model_B import TMModelBInterface
from .TM_Model_C import TMModelCInterface
from .TM_Model_D import TMModelDInterface


def _get_model_dict() -> dict:
    return {
        'TM_Model_A': TMModelAInterface,
        'TM_Model_B': TMModelBInterface,
        'TM_Model_C': TMModelCInterface,
        'TM_Model_D': TMModelDInterface,
    }


def model_zoo(args: object, model_name: str) -> object:
    model_dict = _get_model_dict()
    assert model_name in model_dict, (
        f"Unknown --modelName {model_name!r}. Registered: {list(model_dict)}"
    )
    return model_dict[model_name](args)
