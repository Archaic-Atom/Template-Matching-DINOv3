# -*- coding: utf-8 -*
import JackFramework as jf
from .TM_Model_A import TMModelAInterface


def _get_model_dict() -> dict:
    return {'TM_Model_A': TMModelAInterface,
            }


def model_zoo(args: object, model_name: str) -> object:
    model_dict = _get_model_dict()
    assert model_name in model_dict
    return model_dict[model_name](args)
