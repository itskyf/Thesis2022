import subprocess
from pathlib import Path
from typing import List

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name: str, module: str, sources: List[str]):
    module_path = module.replace(".", "/")
    return CUDAExtension(
        name=f"{module}.{name}", sources=[f"{module_path}/src/{f_name}" for f_name in sources]
    )


if __name__ == "__main__":
    commit_short_hash = "0" * 7
    try:
        commit_short_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        )
    except subprocess.CalledProcessError:
        print("Not found latest commit hash, fallback to", commit_short_hash)
    with Path("pcdet/__init__.py").open("w", encoding="utf-8") as ver_file:
        ver_file.write(f'__version__ = "0.5.2+{commit_short_hash}"')

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
            make_cuda_ext(
                name="votr_ops_cuda",
                module="pcdet.ops.votr_ops",
                sources=[
                    "votr_api.cpp",
                    "build_mapping.cpp",
                    "build_mapping_gpu.cu",
                    "build_attention_indices.cpp",
                    "build_attention_indices_gpu.cu",
                    "group_features.cpp",
                    "group_features_gpu.cu",
                ],
            ),
        ],
    )
