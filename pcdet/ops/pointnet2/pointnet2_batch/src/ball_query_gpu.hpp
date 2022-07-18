#pragma once

#include <ATen/core/Tensor.h>

at::Tensor ball_query(const at::Tensor &points, const at::Tensor &centroids, const int n_neighbors,
                      const float radius);
