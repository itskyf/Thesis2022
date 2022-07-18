#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ball_query_gpu.h"
#include "group_points_gpu.h"
#include "interpolate_gpu.h"
#include "sampling_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query_wrapper", &ball_query_wrapper_fast, "ball_query_wrapper_fast");

  m.def("group_points_wrapper", &group_points_wrapper_fast, "group_points_wrapper_fast");
  m.def("group_points_grad_wrapper", &group_points_grad_wrapper_fast,
        "group_points_grad_wrapper_fast");

  m.def("gather_points_wrapper", &gather_points_wrapper_fast, "gather_points_wrapper_fast");
  m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast,
        "gather_points_grad_wrapper_fast");

  m.def("farthest_point_sampling", &farthest_point_sampling, "Farthest point sampling (CUDA)");
}
