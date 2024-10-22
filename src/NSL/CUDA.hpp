#ifndef NSL_CUDA_HPP
#define NSL_CUDA_HPP

#ifdef USE_CUDA

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>

void stream_sync(
    at::cuda::CUDAStream& dependency,
    at::cuda::CUDAStream& dependent) {
  at::cuda::CUDAEvent cuda_ev;
  cuda_ev.record(dependency);
  cuda_ev.block(dependent);
}

#endif // USE_CUDA

#endif // NSL_CUDA_HPP