from typing import List

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name: str, module: str, sources: List[str]):
    module_path = module.replace(".", "/")
    return CUDAExtension(
        name=f"{module}.{name}", sources=[f"{module_path}/src/{f_name}" for f_name in sources]
    )


if __name__ == "__main__":
    setuptools.setup(
        cmdclass={
            "build_ext": BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name="iou3d_nms_cuda",
                module="pcdet.ops.iou3d_nms",
                sources=[
                    "iou3d_cpu.cpp",
                    "iou3d_nms_api.cpp",
                    "iou3d_nms.cpp",
                    "iou3d_nms_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_cuda",
                module="pcdet.ops.roiaware_pool3d",
                sources=[
                    "roiaware_pool3d.cpp",
                    "roiaware_pool3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="roipoint_pool3d_cuda",
                module="pcdet.ops.roipoint_pool3d",
                sources=[
                    "roipoint_pool3d.cpp",
                    "roipoint_pool3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2_stack_cuda",
                module="pcdet.ops.pointnet2.pointnet2_stack",
                sources=[
                    "ball_query.cpp",
                    "ball_query_gpu.cu",
                    "group_points.cpp",
                    "group_points_gpu.cu",
                    "interpolate.cpp",
                    "interpolate_gpu.cu",
                    "pointnet2_api.cpp",
                    "sampling.cpp",
                    "sampling_gpu.cu",
                    "vector_pool.cpp",
                    "vector_pool_gpu.cu",
                    "voxel_query.cpp",
                    "voxel_query_gpu.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2_batch_cuda",
                module="pcdet.ops.pointnet2.pointnet2_batch",
                sources=[
                    "ball_query.cpp",
                    "ball_query_gpu.cu",
                    "group_points.cpp",
                    "group_points_gpu.cu",
                    "interpolate.cpp",
                    "interpolate_gpu.cu",
                    "pointnet2_api.cpp",
                    "sampling.cpp",
                    "sampling_gpu.cu",
                ],
            ),
        ],
    )
