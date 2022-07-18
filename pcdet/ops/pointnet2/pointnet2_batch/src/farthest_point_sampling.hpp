#pragma once

#include <ATen/core/Tensor.h>

at::Tensor farthest_point_sampling(const at::Tensor &points, const int n_sample);
