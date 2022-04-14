/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void YZHReluForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : 0; }
}

}  // namespace

template<typename T>
class GpuYZHReluKernel final : public user_op::OpKernel {
 public:
  GpuYZHReluKernel() = default;
  ~GpuYZHReluKernel() = default;
 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    RUN_CUDA_KERNEL((YZHReluForwardGpu<T>), ctx->stream(), elem_cnt, elem_cnt,
                    x->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_YZH_RELU_KERNEL(dtype)             \
  REGISTER_USER_KERNEL("yzh_relu")                      \
      .SetCreateFn<GpuYZHReluKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); // 这一行文档里写的有问题会导致编译不过

REGISTER_GPU_YZH_RELU_KERNEL(float)
REGISTER_GPU_YZH_RELU_KERNEL(double)
REGISTER_GPU_YZH_RELU_KERNEL(uint8_t)
REGISTER_GPU_YZH_RELU_KERNEL(int8_t)
REGISTER_GPU_YZH_RELU_KERNEL(int32_t)
REGISTER_GPU_YZH_RELU_KERNEL(int64_t)

}  // namespace oneflow
