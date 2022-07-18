#include <torch/extension.h>

#include "ball_query_gpu.hpp"
#include "farthest_point_sampling.hpp"
#include "group_points_gpu.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &ball_query, "Ball query (GPU)");

  m.def("group_points", &group_points, "Group points (GPU)");
  m.def("group_points_backward", &group_points_backward, "Group points backward (CUDA)");

  m.def("farthest_point_sampling", &farthest_point_sampling, "Farthest point sampling (CUDA)");
}
