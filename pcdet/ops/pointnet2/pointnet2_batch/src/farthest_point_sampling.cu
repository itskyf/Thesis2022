#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAException.h>

#include "cuda_utils.h"

#include "farthest_point_sampling.hpp"

__device__ void __update(const int idx1, const int idx2, float *__restrict__ dists,
                         int *__restrict__ dists_i) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void farthest_point_sampling_kernel(const float *__restrict__ dataset,
                                               float *__restrict__ temp, const int b, const int n,
                                               const int m, int *__restrict__ idxs) {
  // dataset: (B, N, 3)
  // tmp: (B, N)
  // output:
  //      idx: (B, M)

  if (m <= 0) {
    return;
  }
  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  const int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;

  const int tid = threadIdx.x, stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) {
    idxs[0] = old;
  }

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      // if (mag <= 1e-3)
      // continue;

      float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
      if (tid < 512) {
        __update(tid, tid + 512, dists, dists_i);
      }
      __syncthreads();
    }

    if (block_size >= 512) {
      if (tid < 256) {
        __update(tid, tid + 256, dists, dists_i);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update(tid, tid + 128, dists, dists_i);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update(tid, tid + 64, dists, dists_i);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update(tid, tid + 32, dists, dists_i);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update(tid, tid + 16, dists, dists_i);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update(tid, tid + 8, dists, dists_i);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update(tid, tid + 4, dists, dists_i);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update(tid, tid + 2, dists, dists_i);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update(tid, tid + 1, dists, dists_i);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) {
      idxs[j] = old;
    }
  }
}

at::Tensor farthest_point_sampling(const at::Tensor &points, const int n_sample) {
  TORCH_CHECK(points.is_cuda(), "Only support CUDA tensor");
  TORCH_CHECK(points.dim() == 3, "Only support 3D points");
  const int batch_size = points.size(0), total_pts = points.size(1);
  TORCH_CHECK(n_sample <= total_pts, "Not enough input points");

  auto pts_ops = points.options();
  at::Tensor idxs = at::empty({batch_size, n_sample}, pts_ops.dtype(at::kInt));
  at::Tensor tmp = at::full({batch_size, total_pts}, 1e10, pts_ops);

  const at::Tensor c_pts = points.contiguous();
  const float *p_pts = c_pts.data_ptr<float>();
  float *p_tmp = tmp.data_ptr<float>();
  int *p_idxs = idxs.data_ptr<int>();

  unsigned int n_threads = opt_n_threads(total_pts);
  switch (n_threads) {
    case 1024:
      farthest_point_sampling_kernel<1024>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 512:
      farthest_point_sampling_kernel<512>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 256:
      farthest_point_sampling_kernel<256>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 128:
      farthest_point_sampling_kernel<128>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 64:
      farthest_point_sampling_kernel<64>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 32:
      farthest_point_sampling_kernel<32>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 16:
      farthest_point_sampling_kernel<16>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 8:
      farthest_point_sampling_kernel<8>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 4:
      farthest_point_sampling_kernel<4>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 2:
      farthest_point_sampling_kernel<2>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    case 1:
      farthest_point_sampling_kernel<1>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
      break;
    default:
      farthest_point_sampling_kernel<512>
          <<<batch_size, n_threads>>>(p_pts, p_tmp, batch_size, total_pts, n_sample, p_idxs);
  }
  AT_CUDA_CHECK(cudaGetLastError());

  return idxs;
}
