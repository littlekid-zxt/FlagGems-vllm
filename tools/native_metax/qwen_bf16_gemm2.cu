#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "mctlass/epilogue/thread/maca_linear_combination_moe.h"
#include "mctlass/frontend_op/gemm_config.h"

namespace {

using Element = mctlass::bfloat16_t;
using RowMajor = mctlass::layout::RowMajor;
using ColumnMajor = mctlass::layout::ColumnMajor;
using ThreadblockShape = mctlass::gemm::GemmShape<128, 128, 64>;
using WarpShape = mctlass::gemm::GemmShape<32, 128, 64>;
using InstructionShape = mctlass::gemm::GemmShape<16, 16, 16>;
using EpilogueOutputOp = mctlass::epilogue::thread::MacaLinearCombinationMoe<
    float, 2, float, Element, true>;
using GemmKernel = typename gemmMoeConfig<
    Element,
    RowMajor,
    Element,
    ColumnMajor,
    Element,
    RowMajor,
    float,
    mctlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    1,
    32,
    128,
    true>::GemmKernel;
using Mma = typename GemmKernel::Mma;

__global__ void flaggems_qwen_bf16_gemm2_kernel(
    void const* a,
    void const* b,
    int num_valid_tokens,
    int n,
    int k,
    MoeParams moe_params) {
  int thread_idx = threadIdx.x;
  Mma mma(thread_idx, thread_idx / 64, thread_idx % 64);
  int num_tokens_post_padded;
  int group_idx;
  mma(
      a,
      b,
      num_valid_tokens,
      n,
      k,
      num_valid_tokens,
      moe_params,
      &num_tokens_post_padded,
      &group_idx,
      nullptr,
      nullptr,
      nullptr);
}

void qwen_bf16_gemm2(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    bool full_kernel,
    int grid_x_override,
    int topk) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda());
  TORCH_CHECK(a.scalar_type() == at::kBFloat16);
  TORCH_CHECK(b.scalar_type() == at::kBFloat16);
  TORCH_CHECK(c.scalar_type() == at::kBFloat16);
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat);
  TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt);
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt);
  TORCH_CHECK(num_tokens_post_padded.scalar_type() == at::kInt);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 3);
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous());
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(sorted_token_ids.is_contiguous());
  TORCH_CHECK(expert_ids.is_contiguous());
  TORCH_CHECK(num_tokens_post_padded.is_contiguous());
  TORCH_CHECK(b.size(1) % 4 == 0 && b.size(2) % 8 == 0);
  TORCH_CHECK(topk > 0 && (topk & (topk - 1)) == 0);

  int num_valid_tokens = a.size(0) * topk;
  int num_experts = b.size(0);
  int n = b.size(1);
  int k = b.size(2);
  int em = sorted_token_ids.size(0);
  TORCH_CHECK(topk_weights.numel() == num_valid_tokens);
  TORCH_CHECK(c.numel() == num_valid_tokens * n);

  MoeParams moe_params(
      sorted_token_ids.data_ptr<int>(),
      expert_ids.data_ptr<int>(),
      num_tokens_post_padded.data_ptr<int>(),
      em,
      topk,
      true,
      topk_weights.data_ptr(),
      c.data_ptr());

  int grid_m = (em + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
  int group_blocks = std::max(
      1,
      std::min(
          8,
          (num_valid_tokens / num_experts + ThreadblockShape::kM - 1) /
              ThreadblockShape::kM));
  int grid_x = std::min(grid_m, group_blocks);
  if (grid_x_override > 0) {
    grid_x = std::min(grid_m, grid_x_override);
  }
  int grid_z = (grid_m + grid_x - 1) / grid_x;
  int grid_y = (n + ThreadblockShape::kN - 1) / ThreadblockShape::kN;
  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(256, 1, 1);

  c10::cuda::CUDAGuard device_guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  if (full_kernel) {
    typename GemmKernel::Arguments arguments(
        mctlass::gemm::GemmUniversalMode::kGemm,
        mctlass::gemm::BatchedGemmCoord(
            num_valid_tokens, n, k, num_experts),
        typename EpilogueOutputOp::Params(topk_weights.data_ptr<float>()),
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        moe_params);
    typename GemmKernel::Params params(arguments);
    mctlass::Kernel<GemmKernel><<<grid, block, 0, stream>>>(params);
  } else {
    flaggems_qwen_bf16_gemm2_kernel<<<grid, block, 0, stream>>>(
        a.data_ptr(),
        b.data_ptr(),
        num_valid_tokens,
        n,
        k,
        moe_params);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qwen_bf16_gemm2", &qwen_bf16_gemm2);
}
