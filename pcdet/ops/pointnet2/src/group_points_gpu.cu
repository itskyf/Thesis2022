/*
batch version of point grouping, modified from the original implementation of official PointNet++
codes. Written by Shaoshuai Shi All Rights Reserved 2018.
*/

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "cuda_utils.h"

#include "group_points_gpu.hpp"

__global__ void group_points_kernel(const float *__restrict__ p_points,
                                    const int *__restrict__ p_idxs, const int batch_size,
                                    const int n_channels, const int total_pts, const int n_groups,
                                    const int n_neighbors, float *__restrict__ p_grouped) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
  const int bs_idx = blockIdx.z, c_idx = blockIdx.y;
  const int index = blockIdx.x * blockDim.x + threadIdx.x, pt_idx = index / n_neighbors;
  if (bs_idx >= batch_size || c_idx >= n_channels || pt_idx >= n_groups) {
    return;
  }
  const int sample_idx = index % n_neighbors;
  p_idxs += bs_idx * n_groups * n_neighbors + pt_idx * n_neighbors + sample_idx;

  const int in_idx = bs_idx * n_channels * total_pts + c_idx * total_pts + p_idxs[0];
  const int out_idx = bs_idx * n_channels * n_groups * n_neighbors +
                      c_idx * n_groups * n_neighbors + pt_idx * n_neighbors + sample_idx;

  p_grouped[out_idx] = p_points[in_idx];
}

torch::Tensor group_points(const torch::Tensor &points, const torch::Tensor &indices,
                           const int total_pts) {
  TORCH_CHECK(points.is_cuda() && indices.is_cuda(), "Only supports CUDA tensor");
  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

  const int batch_size = points.size(0), n_channels = points.size(1);
  TORCH_CHECK(batch_size == indices.size(0), "Batch size must be the same");
  const int n_groups = indices.size(1), n_neighbors = indices.size(2);

  torch::Tensor grouped =
      torch::empty({batch_size, n_channels, n_groups, n_neighbors}, points.options());

  const at::Tensor c_points = points.contiguous();
  const float *p_points = c_points.data_ptr<float>();
  const int *p_idxs = indices.data_ptr<int>();
  float *p_grouped = grouped.data_ptr<float>();

  dim3 blocks(DIVUP(n_groups * n_neighbors, THREADS_PER_BLOCK), n_channels, batch_size);
  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_kernel<<<blocks, threads>>>(p_points, p_idxs, batch_size, n_channels, total_pts,
                                           n_groups, n_neighbors, p_grouped);
  AT_CUDA_CHECK(cudaGetLastError());
  return grouped;
}
__global__ void group_points_backward_kernel(const int *__restrict__ p_idxs, const int batch_size,
                                             const int n_channels, const int total_pts,
                                             const int n_groups, const int n_neighbors,
                                             float *__restrict__ p_grad_grouped,
                                             float *__restrict__ p_grad_feats) {
  // p_grad_grouped: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)
  const int bs_idx = blockIdx.z, c_idx = blockIdx.y;
  const int index = blockIdx.x * blockDim.x + threadIdx.x, pt_idx = index / n_neighbors;
  if (bs_idx >= batch_size || c_idx >= n_channels || pt_idx >= n_groups) {
    return;
  }
  const int sample_idx = index % n_neighbors;

  p_grad_grouped += bs_idx * n_channels * n_groups * n_neighbors + c_idx * n_groups * n_neighbors +
                    pt_idx * n_neighbors + sample_idx;
  p_idxs += bs_idx * n_groups * n_neighbors + pt_idx * n_neighbors + sample_idx;
  atomicAdd(p_grad_feats + bs_idx * n_channels * total_pts + c_idx * total_pts + p_idxs[0],
            p_grad_grouped[0]);
}

torch::Tensor group_points_backward(torch::Tensor &grad_grouped, const torch::Tensor &indices,
                                    const int total_pts) {
  TORCH_CHECK(indices.is_contiguous(), "indices is not contiguous");
  const int batch_size = grad_grouped.size(0), n_channels = grad_grouped.size(1),
            n_groups = grad_grouped.size(2), n_neighbors = grad_grouped.size(3);

  torch::Tensor grad_feats =
      torch::zeros({batch_size, n_channels, total_pts}, grad_grouped.options().requires_grad(true));

  grad_grouped = grad_grouped.contiguous();
  const int *p_idxs = indices.data_ptr<int>();
  float *p_grad_grouped = grad_grouped.data_ptr<float>();
  float *p_grad_feats = grad_feats.data_ptr<float>();

  dim3 blocks(DIVUP(n_groups * n_neighbors, THREADS_PER_BLOCK), n_channels, batch_size);
  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  group_points_backward_kernel<<<blocks, threads>>>(p_idxs, batch_size, n_channels, total_pts,
                                                    n_groups, n_neighbors, p_grad_grouped,
                                                    p_grad_feats);
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_feats;
}
