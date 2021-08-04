#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template <typename T>
void Logical_not(DeviceCtx *ctx, const int64_t n, const T *x, T *y) {
  for (int64_t i = 0; i != n; ++i) {
    if((x[i]==true)||(x[i]!=0)){
      y[i]=false;
    }else if((x[i]==false)||x[i]==0){
      y[i]=true;
    }
  }
}

template <DeviceType device_type, typename T>
class Logical_not_Kernel final : public user_op::OpKernel {
public:
  Logical_not_Kernel() = default;
  ~Logical_not_Kernel() = default;

private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    Logical_not<T>(ctx->device_ctx(),
           in_tensor->shape().elem_cnt(),
           in_tensor->dptr<T>(),
           out_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("logical_not")                     \
      .SetCreateFn<Logical_not_Kernel<device, dtype>>()      \
      .SetIsMatchedHob(                              \
          (user_op::HobDeviceTag() == device) &     \
          (user_op::HobDataType("out", 0)            \
            == GetDataType<dtype>::value));

REGISTER_RELU_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_RELU_KERNEL(DeviceType::kGPU, int8_t)
} // namespace

} // namespace oneflow
