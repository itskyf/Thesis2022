from .caddn import CaDDN
from .centerpoint import CenterPoint
from .ct3d import CT3D
from .ct3d_3cat import CT3D_3CAT
from .detector3d_template import Detector3DTemplate
from .parta2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .votr_ssd_net import VoTrSSD
from .votr_tsd_net import VoTrRCNN
from .voxel_rcnn import VoxelRCNN

__all__ = {
    "CaDDN": CaDDN,
    "CenterPoint": CenterPoint,
    "CT3D_3CAT": CT3D_3CAT,
    "CT3D": CT3D,
    "Detector3DTemplate": Detector3DTemplate,
    "PartA2Net": PartA2Net,
    "PointPillar": PointPillar,
    "PointRCNN": PointRCNN,
    "PVRCNNPlusPlus": PVRCNNPlusPlus,
    "PVRCNN": PVRCNN,
    "SECONDNetIoU": SECONDNetIoU,
    "SECONDNet": SECONDNet,
    "VoTrRCNN": VoTrRCNN,
    "VoTrSSD": VoTrSSD,
    "VoxelRCNN": VoxelRCNN,
    "VoxelRCNN": VoxelRCNN,
}


def build_detector(model_cfg, num_class, dataset) -> Detector3DTemplate:
    return __all__[model_cfg.NAME](model_cfg=model_cfg, num_class=num_class, dataset=dataset)
