#pragma once

#include <ATen/core/Tensor.h>

at::Tensor group_points(const at::Tensor& points, const at::Tensor& idxs);

at::Tensor group_points_backward(const at::Tensor& grad_grouped, const at::Tensor& idxs,
                                 const int total_pts);
