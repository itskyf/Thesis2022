from .ct3d_head import CT3DHead
from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate
from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .voxelrcnn_trans_head import VoxelRCNNHeadTrans

__all__ = {
    "CT3DHead": CT3DHead,
    "PartA2FCHead": PartA2FCHead,
    "PointRCNNHead": PointRCNNHead,
    "PVRCNNHead": PVRCNNHead,
    "RoIHeadTemplate": RoIHeadTemplate,
    "SECONDHead": SECONDHead,
    "VoxelRCNNHead": VoxelRCNNHead,
    "VoxelRCNNTrans": VoxelRCNNHeadTrans
}
