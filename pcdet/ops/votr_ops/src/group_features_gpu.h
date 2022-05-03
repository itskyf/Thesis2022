/*
Modified from group points, don't care indices with -1(<0)
Written by Jiageng Mao
All Rights Reserved 2019-2020.
*/

#ifndef _STACK_GROUP_FEATURES_GPU_H
#define _STACK_GROUP_FEATURES_GPU_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/serialize/tensor.h>
#include <vector>

int group_features_wrapper_stack(int B, int M, int C, int nsample,
                                 at::Tensor features_tensor,
                                 at::Tensor features_batch_cnt_tensor,
                                 at::Tensor idx_tensor,
                                 at::Tensor idx_batch_cnt_tensor,
                                 at::Tensor out_tensor);

void group_features_kernel_launcher_stack(int B, int M, int C, int nsample,
                                          const float *features,
                                          const int *features_batch_cnt,
                                          const int *idx,
                                          const int *idx_batch_cnt, float *out);

int group_features_grad_wrapper_stack(int B, int M, int C, int N, int nsample,
                                      at::Tensor grad_out_tensor,
                                      at::Tensor idx_tensor,
                                      at::Tensor idx_batch_cnt_tensor,
                                      at::Tensor features_batch_cnt_tensor,
                                      at::Tensor grad_features_tensor);

void group_features_grad_kernel_launcher_stack(
    int B, int M, int C, int N, int nsample, const float *grad_out,
    const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt,
    float *grad_features);

#endif
