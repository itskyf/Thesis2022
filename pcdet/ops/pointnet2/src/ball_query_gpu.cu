/*
batch version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/
#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>

#include "cuda_utils.h"

#include "ball_query_gpu.hpp"

__global__ void ball_query_kernel(const float *__restrict__ p_pts,
                                  const float *__restrict__ p_centroids, const int n_neighbors,
                                  const float radius, const int batch_size, const int total_pts,
                                  const int n_centroids, int *__restrict__ p_idxs) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  const int bs_idx = blockIdx.y;
  const int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= batch_size || pt_idx >= n_centroids) {
    return;
  }

  p_centroids += bs_idx * n_centroids * 3 + pt_idx * 3;
  p_pts += bs_idx * total_pts * 3;
  p_idxs += bs_idx * n_centroids * n_neighbors + pt_idx * n_neighbors;

  float radius2 = radius * radius;
  float new_x = p_centroids[0], new_y = p_centroids[1], new_z = p_centroids[2];

  int cnt = 0;
  for (int k = 0; k < total_pts; ++k) {
    const float x = p_pts[k * 3 + 0], y = p_pts[k * 3 + 1], z = p_pts[k * 3 + 2];
    const float d2 =
        (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
    if (d2 < radius2) {
      if (cnt == 0) {
        for (int l = 0; l < n_neighbors; ++l) {
          p_idxs[l] = k;
        }
      }
      p_idxs[cnt] = k;
      ++cnt;
      if (cnt >= n_neighbors) {
        break;
      }
    }
  }
}

at::Tensor ball_query(const at::Tensor &points, const at::Tensor &centroids, const int n_neighbors,
                      const float radius) {
  TORCH_CHECK(points.is_cuda() && centroids.is_cuda(), "Only support CUDA tensor");
  TORCH_CHECK(points.dim() == 3 && centroids.dim() == 3, "Only support 3D points");

  const int batch_size = points.size(0), total_pts = points.size(1);
  const int n_centroids = centroids.size(1);
  TORCH_CHECK(batch_size == centroids.size(0),
              "points and centroids must have the same batch dimension");
  TORCH_CHECK(n_centroids <= total_pts, "Not enough input points");

  at::Tensor idxs =
      at::zeros({batch_size, n_centroids, n_neighbors}, points.options().dtype(at::kInt));

  const at::Tensor c_pts = points.contiguous(), c_centroids = centroids.contiguous();
  const float *p_pts = c_pts.data_ptr<float>();
  const float *p_centroids = c_centroids.data_ptr<float>();
  int *p_idxs = idxs.data_ptr<int>();

  dim3 blocks(DIVUP(n_centroids, THREADS_PER_BLOCK), batch_size);
  dim3 threads(THREADS_PER_BLOCK);
  ball_query_kernel<<<blocks, threads>>>(p_pts, p_centroids, n_neighbors, radius, batch_size,
                                         total_pts, n_centroids, p_idxs);
  // cudaDeviceSynchronize();  // for using printf in kernel function
  AT_CUDA_CHECK(cudaGetLastError());

  return idxs;
}
