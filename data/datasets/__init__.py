# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID, DukeMTMCreIDPartial, DukeMTMCreIDMIXq
from .market1501 import Market1501, Market1501Partial
from .msmt17 import MSMT17
from .veri import VeRi
from .dataset_loader import ImageDataset, ImageDatasetEma
from .imagenet import ImageNet
from .partial_iLIDS import Partial_iLIDS
from .partial_REID import Partial_REID
from .partial_REID_group import Partial_REID_group
# from .market1501_partial import Market1501Partial

__factory = {
    'market1501': Market1501,
    'market1501_partial': Market1501Partial,
    # 'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'dukemtmc_mix': DukeMTMCreIDMIXq,
    'dukemtmc_partial': DukeMTMCreIDPartial,
    'msmt17': MSMT17,
    'veri': VeRi,
    'imagenet': ImageNet,
    'partial_reid': Partial_REID,
    'partial_ilids': Partial_iLIDS
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
