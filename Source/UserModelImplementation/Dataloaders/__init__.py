# -*- coding: utf-8 -*-
import JackFramework as jf
from .tm_dataloader import TMDataloader


def _get_dataloaders_dict() -> dict:
    return {'tmdataset': TMDataloader}


def dataloaders_zoo(args: object, dataset_name: str) -> object:
    dataloader_dict = _get_dataloaders_dict()
    assert dataset_name in dataloader_dict
    return dataloader_dict[dataset_name](args)
