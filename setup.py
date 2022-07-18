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
    COMMIT_SHORT_HASH = "0" * 7
    try:
        COMMIT_SHORT_HASH = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        )
    except FileNotFoundError:
        print("Not found latest commit hash, fallback to", COMMIT_SHORT_HASH)
    with Path("pcdet/__init__.py").open("w", encoding="utf-8") as ver_file:
        ver_file.write(f'__version__ = "0.5.2+{COMMIT_SHORT_HASH}"')

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
                name="pointnet2_batch_cuda",
                module="pcdet.ops.pointnet2.pointnet2_batch",
                sources=[
                    "ball_query.cpp",
                    "ball_query_gpu.cu",
                    "group_points.cpp",
                    "group_points_gpu.cu",
                    "pointnet2_api.cpp",
                    "sampling_gpu.cu",
                ],
            ),
        ],
    )
