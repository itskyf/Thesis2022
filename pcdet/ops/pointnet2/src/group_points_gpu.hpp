#pragma once

#include <torch/extension.h>

torch::Tensor group_points(const torch::Tensor& points, const torch::Tensor& indices,
                           const int total_pts);

torch::Tensor group_points_backward(torch::Tensor& grad_grouped, const torch::Tensor& indices,
                                    const int total_pts);
