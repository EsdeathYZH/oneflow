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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/cuda_stream_type.h"
#include "oneflow/core/eager/opkernel_instruction_type.h"
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job/parallel_signature.cfg.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {
namespace vm {

struct LocalCallOpKernelUtil final {
  static inline Maybe<void> Compute(const vm::InstructionMsg& instr_msg) {
    OF_PROFILER_RANGE_PUSH("ResetPrior");
    auto* operand = LocalCallOpKernelUtil::GetLocalCallOpKernelPhyInstrOperand(instr_msg);
    operand->mut_opkernel()->composed_attrs_for_scheduler_thread()->ResetPrior(operand->attrs());
    DeviceCtx* device_ctx = instr_msg.phy_instr_stream()->device_ctx().get();
    OF_PROFILER_RANGE_POP();
    OF_PROFILER_RANGE_PUSH("AllocateOutputBlobsMemory");
    JUST(AllocateOutputBlobsMemory(operand, device_ctx));
    OF_PROFILER_RANGE_POP();
    if (unlikely(operand->need_temp_storage())) {
      OF_PROFILER_RANGE_PUSH("TryAllocateTempStorageBlobMemory");
      InferTempStorageBlobDesc(operand);
      JUST(ResetTempStorageBlob(operand));
      JUST(TryAllocateTempStorageBlobMemory(operand, device_ctx));
      OF_PROFILER_RANGE_POP();
    }
    user_op::OpKernelState* state = nullptr;
    user_op::OpKernelCache* cache = nullptr;
    if (operand->user_opkernel()->has_state_or_cache()) {
      OF_PROFILER_RANGE_PUSH("TryInitOpKernelStateAndCache");
      TryInitOpKernelStateAndCache(operand, device_ctx, &state, &cache);
      OF_PROFILER_RANGE_POP();
    }
    OpKernelCompute(operand, device_ctx, state, cache);
    if (unlikely(operand->need_temp_storage())) {
      OF_PROFILER_RANGE_PUSH("DeallocateTempStorageBlobMemory");
      JUST(DeallocateTempStorageBlobMemory(operand, device_ctx));
      OF_PROFILER_RANGE_POP();
    }
    return Maybe<void>::Ok();
  }

  static inline LocalCallOpKernelPhyInstrOperand* GetLocalCallOpKernelPhyInstrOperand(
      const vm::InstructionMsg& instr_msg) {
    auto* operand = CHECK_NOTNULL(instr_msg.phy_instr_operand().get());
    return CHECK_NOTNULL(dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand));
  }

 private:
  static inline void InferTempStorageBlobDesc(LocalCallOpKernelPhyInstrOperand* operand) {
    const auto& InferTmpSizeFn = operand->opkernel().GetInferTmpSizeFn(operand->user_opkernel());
    auto* temp_blob_desc = operand->mut_opkernel()->mut_temp_blob_object()->mut_blob_desc();
    CHECK(temp_blob_desc->data_type() == DataType::kChar);
    one::LocalUserOpInferContext* op_infer_ctx =
        operand->opkernel().op_infer_ctx_for_scheduler_thread();
    op_infer_ctx->Update(operand->inputs().get(), operand->outputs().get(),
                         operand->consistent_tensor_infer_result().get());
    size_t temp_size = InferTmpSizeFn(op_infer_ctx);
    temp_blob_desc->mut_shape() = Shape({static_cast<int64_t>(temp_size)});
    temp_blob_desc->set_is_dynamic(true);
    op_infer_ctx->Update(nullptr, nullptr, nullptr);
  }

  static inline Maybe<void> ResetTempStorageBlob(LocalCallOpKernelPhyInstrOperand* operand) {
    return operand->mut_opkernel()->mut_temp_blob_object()->InitBlob();
  }

  static inline void TryInitOpKernelStateAndCache(LocalCallOpKernelPhyInstrOperand* operand,
                                                  DeviceCtx* device_ctx,
                                                  user_op::OpKernelState** state,
                                                  user_op::OpKernelCache** cache) {
    if (likely(operand->op_interp_ctx().state)) {
      *state = operand->op_interp_ctx().state.get();
      // set state to nullptr so that state initialization in TryInitOpKernelStateAndCache will be
      // skipped.
      state = nullptr;
    }
    operand->mut_opkernel()->TryInitOpKernelStateAndCache(
        operand->user_opkernel(), device_ctx, operand->inputs().get(), operand->outputs().get(),
        operand->consistent_tensor_infer_result().get(), state, cache);
  }

  static inline Maybe<void> AllocateOutputBlobsMemory(LocalCallOpKernelPhyInstrOperand* operand,
                                                      DeviceCtx* device_ctx) {
    for (const auto& blob_object : *operand->outputs()) {
      JUST(blob_object->TryInitBlob());
      JUST(blob_object->TryAllocateBlobBodyMemory(device_ctx));
    }
    return Maybe<void>::Ok();
  }

  static inline Maybe<void> TryAllocateTempStorageBlobMemory(
      LocalCallOpKernelPhyInstrOperand* operand, DeviceCtx* device_ctx) {
    return operand->mut_opkernel()->mut_temp_blob_object()->TryAllocateBlobBodyMemory(device_ctx);
  }

  static inline void OpKernelCompute(LocalCallOpKernelPhyInstrOperand* operand,
                                     DeviceCtx* device_ctx, user_op::OpKernelState* state,
                                     const user_op::OpKernelCache* cache) {
    auto* opkernel = operand->mut_opkernel();
    auto* compute_ctx =
        opkernel->UpdateComputeContext(operand->inputs().get(), operand->outputs().get(),
                                       operand->consistent_tensor_infer_result().get(), device_ctx);
    OF_PROFILER_RANGE_PUSH("Compute");
    operand->user_opkernel()->Compute(compute_ctx, state, cache);
    OF_PROFILER_RANGE_POP();
    // tensor tuples are not allowed to be hold by StatefulLocalOpKernel
    opkernel->UpdateComputeContext(nullptr, nullptr, nullptr, nullptr);
  }

  static inline Maybe<void> DeallocateTempStorageBlobMemory(
      LocalCallOpKernelPhyInstrOperand* operand, DeviceCtx* device_ctx) {
    return operand->mut_opkernel()->mut_temp_blob_object()->DeallocateBlobDataPtr();
  }
};

void LocalCallOpKernelInstructionType::Compute(vm::Instruction* instruction) const {
  CHECK_JUST(LocalCallOpKernelUtil::Compute(instruction->instr_msg()));
}

void LocalCallOpKernelInstructionType::ComputeInFuseMode(vm::InstructionMsg* instr_msg) const {
  CHECK_JUST(LocalCallOpKernelUtil::Compute(*instr_msg));
}

std::string LocalCallOpKernelInstructionType::DebugOpTypeName(
    const vm::InstructionMsg& instr_msg) const {
  auto* operand = CHECK_NOTNULL(instr_msg.phy_instr_operand().get());
  return CHECK_NOTNULL(dynamic_cast<LocalCallOpKernelPhyInstrOperand*>(operand))
      ->opkernel()
      .op_type_name();
}

}  // namespace vm
}  // namespace oneflow
