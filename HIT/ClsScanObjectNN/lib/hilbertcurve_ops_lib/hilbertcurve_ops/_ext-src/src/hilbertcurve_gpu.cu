#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

typedef int32_t point_type;
typedef int64_t distance_type;

// input : points, (b_s, n_p, d_p)
// output : distances, (b_s, n_p, 1)
__global__ void distances_from_points_kernel(int b_s, int n_p, int d_p, int m_l,
                      point_type *__restrict__ points, distance_type *__restrict__ distances) {
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b_s || pt_idx >= n_p) return;

  point_type * points_i = points + (bs_idx * n_p + pt_idx) * d_p;
  distance_type * distances_i = distances + bs_idx * n_p + pt_idx;
  int M = 1 << (m_l - 1);
  int q = M, t, p;
  distance_type dis = 0;
  while (q > 1) {
    p = q - 1;
    for (int i_d = 0; i_d < d_p; i_d++) {
      if (points_i[i_d] & q)
        points_i[0] ^= p;
      else {
        t = (points_i[0] ^ points_i[i_d]) & p;
        points_i[0] ^= t;
        points_i[i_d] ^= t;
      }
    }
    q >>= 1;
  }
  for (int i_d = 1; i_d < d_p; i_d++)
    points_i[i_d] ^= points_i[i_d - 1];
  t = 0, q = M;
  while (q > 1) {
    if (points_i[d_p - 1] & q)
      t ^= q - 1;
    q >>= 1;
  }
  for (int i_d = 0; i_d < d_p; i_d++)
    points_i[i_d] ^= t;
  for (int i_l = m_l - 1; i_l >= 0; i_l--)
    for (int i_d = 0; i_d < d_p; i_d++)
      dis = dis * 2 + ((points_i[i_d] >> i_l) & 1);
  distances_i[0] = dis;
}

void distances_from_points_kernel_wrapper(int b_s, int n_p, int d_p, int m_l,
                                          point_type * points, distance_type * distances) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(DIVUP(n_p, THREADS_PER_BLOCK), b_s);
  dim3 threads(THREADS_PER_BLOCK);

  distances_from_points_kernel<<<blocks, threads, 0, stream>>>(
      b_s, n_p, d_p, m_l, points, distances);

  CUDA_CHECK_ERRORS();
}

// output : distances, (b_s, n_p, 1)
// input  : points, (b_s, n_p, d_p)
__global__ void points_from_distances_kernel(int b_s, int n_p, int d_p, int m_l,
                      distance_type *__restrict__ distances, point_type *__restrict__ points) {
  int bs_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b_s || pt_idx >= n_p) return;

  point_type * points_i = points + (bs_idx * n_p + pt_idx) * d_p;
  distance_type * distances_i = distances + bs_idx * n_p + pt_idx;

  for (int i = d_p * m_l - 1; i >= 0;) {
    for (int i_d = 0; i_d < d_p; i--, i_d++) {
      points_i[i_d] = points_i[i_d] * 2 + ((distances_i[0] >> i) & 1);
    }
  }
  int z = 1 << m_l;
  int t = points_i[d_p - 1] >> 1;
  for (int i_d = d_p - 1; i_d > 0; i_d--)
    points_i[i_d] ^= points_i[i_d -1];
  points_i[0] ^= t;
  int q = 2;
  while (q != z) {
    int p = q - 1;
    for (int i_d = d_p - 1; i_d >= 0; i_d--) {
      if (points_i[i_d] & q)
        points_i[0] ^= p;
      else {
        t = (points_i[0] ^ points_i[i_d]) & p;
        points_i[0] ^= t;
        points_i[i_d] ^= t;
      }
    }
    q <<= 1;
  }
}

void points_from_distances_kernel_wrapper(int b_s, int n_p, int d_p, int m_l,
                                          distance_type * distances, point_type * points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(DIVUP(n_p, THREADS_PER_BLOCK), b_s);
  dim3 threads(THREADS_PER_BLOCK);

  points_from_distances_kernel<<<blocks, threads, 0, stream>>>(
      b_s, n_p, d_p, m_l, distances, points);

  CUDA_CHECK_ERRORS();
}
