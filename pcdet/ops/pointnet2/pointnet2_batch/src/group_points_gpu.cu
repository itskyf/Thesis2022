/*
batch version of point grouping, modified from the original implementation of official PointNet++
codes. Written by Shaoshuai Shi All Rights Reserved 2018.
*/

#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>

#include "cuda_utils.h"

#include "group_points_gpu.hpp"

__global__ void group_points_kernel(const float *__restrict__ p_pts, const int *__restrict__ p_idxs,
                                    const int batch_size, const int n_channels, const int total_pts,
                                    const int n_groups, const int n_neighbors,
                                    float *__restrict__ p_grouped) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
  const int bs_idx = blockIdx.z, c_idx = blockIdx.y, index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pt_idx = index / n_neighbors;
  if (bs_idx >= batch_size || c_idx >= n_channels || pt_idx >= n_groups) {
    return;
  }
  const int sample_idx = index % n_neighbors;

  p_idxs += bs_idx * n_groups * n_neighbors + pt_idx * n_neighbors + sample_idx;

  const int in_idx = bs_idx * n_channels * total_pts + c_idx * total_pts + p_idxs[0];
  const int out_idx = bs_idx * n_channels * n_groups * n_neighbors +
                      c_idx * n_groups * n_neighbors + pt_idx * n_neighbors + sample_idx;
  p_grouped[out_idx] = p_pts[in_idx];
}

at::Tensor group_points(const at::Tensor &points, const at::Tensor &idxs) {
  TORCH_CHECK(points.is_cuda() && idxs.is_cuda(), "Only support CUDA tensor");
  TORCH_CHECK(points.dim() == 3 && idxs.dim() == 4, "Unsupported tensor shape");

  const int batch_size = points.size(0), n_channels = points.size(1), total_pts = points.size(2);
  TORCH_CHECK(batch_size == idxs.size(0), "points and indices must have the save batch dimensions");
  const int n_groups = idxs.size(1), n_neighbors = idxs.size(2);
  TORCH_CHECK(n_groups <= total_pts, "Not enough input points");

  at::Tensor grouped_pts =
      at::empty({batch_size, n_channels, n_groups, n_neighbors}, points.options());

  const at::Tensor c_pts = points.contiguous(), c_idxs = idxs.contiguous();
  const float *p_pts = c_pts.data_ptr<float>();
  const int *p_idxs = c_idxs.data_ptr<int>();
  float *p_grouped = grouped_pts.data_ptr<float>();

  dim3 blocks(DIVUP(n_groups * n_neighbors, THREADS_PER_BLOCK), n_channels, batch_size);
  dim3 threads(THREADS_PER_BLOCK);
  group_points_kernel<<<blocks, threads>>>(p_pts, p_idxs, batch_size, n_channels, total_pts,
                                           n_groups, n_neighbors, p_grouped);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  AT_CUDA_CHECK(cudaGetLastError());

  return grouped_pts;
}

__global__ void group_points_backward_kernel(float *__restrict__ p_g_grouped,
                                             const int *__restrict__ p_idxs, const int batch_size,
                                             const int n_channels, const int total_pts,
                                             const int n_groups, const int n_neighbors,
                                             float *__restrict__ p_out) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      p_g_grouped: (B, C, N)
  const int bs_idx = blockIdx.z, c_idx = blockIdx.y, index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pt_idx = index / n_neighbors;
  if (bs_idx >= batch_size || c_idx >= n_channels || pt_idx >= n_groups) {
    return;
  }
  const int sample_idx = index % n_neighbors;

  p_out += bs_idx * n_channels * n_groups * n_neighbors + c_idx * n_groups * n_neighbors +
           pt_idx * n_neighbors + sample_idx;
  p_idxs += bs_idx * n_groups * n_neighbors + pt_idx * n_neighbors + sample_idx;
  atomicAdd(p_g_grouped + bs_idx * n_channels * total_pts + c_idx * total_pts + p_idxs[0],
            p_out[0]);
}

at::Tensor group_points_backward(const at::Tensor &grad_grouped, const at::Tensor &idxs,
                                 const int total_pts) {
  TORCH_CHECK(grad_grouped.is_cuda() && idxs.is_cuda(), "Only support CUDA tensor");

  const int batch_size = grad_grouped.size(0), n_channels = grad_grouped.size(1),
            n_groups = grad_grouped.size(2), n_neighbors = grad_grouped.size(3);

  at::Tensor grad_feats = at::zeros({batch_size, n_channels, total_pts}, grad_grouped.options());

  const at::Tensor c_grad_grouped = grad_grouped.contiguous(), c_idxs = idxs.contiguous();
  float *p_g_grouped = c_grad_grouped.data_ptr<float>();
  const int *p_idxs = c_idxs.data_ptr<int>();
  float *p_grad_feats = grad_feats.data_ptr<float>();

  dim3 blocks(DIVUP(n_groups * n_neighbors, THREADS_PER_BLOCK), n_channels, batch_size);
  dim3 threads(THREADS_PER_BLOCK);
  group_points_backward_kernel<<<blocks, threads>>>(p_g_grouped, p_idxs, batch_size, n_channels,
                                                    total_pts, n_groups, n_neighbors, p_grad_feats);
  AT_CUDA_CHECK(cudaGetLastError());

  return grad_feats;
}
