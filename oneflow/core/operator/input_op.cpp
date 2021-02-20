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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/input_op.h"
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void InputOp::InitFromOpConf() {
  CHECK(op_conf().has_input_conf());
  if (op_conf().input_conf().has_tick()) { EnrollInputBn("tick", false); }
  OutputBlobModifier* modifier = EnrollOutputBn("out", false);
  modifier->set_is_mutable(true);
  modifier->set_header_infered_before_compute(false);
}

Maybe<void> InputOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  JUST(InterfaceOpUtil::InferOutBlobDesc(op_conf().input_conf().blob_conf(), out_blob_desc,
                                         parallel_ctx));
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = op_conf().input_conf().blob_conf().batch_axis();
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  InterfaceOpUtil::GetInputLikeOpSbpSignature(op_conf().input_conf().blob_conf(), input_bns(),
                                              output_bns(), sbp_signature);
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  InterfaceOpUtil::GetInputLikeOpSbpSignature(op_conf().input_conf().blob_conf(), input_bns(),
                                              output_bns(),
                                              sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferParallelHierarchy(
    std::function<Maybe<const Shape*>(const std::string&)> GetParallelHierarchy4Ibn,
    const ParallelDesc& parallel_desc, Shape* parallel_hierarchy) const {
  const InterfaceBlobConf& blob_conf = op_conf().input_conf().blob_conf();
  if (blob_conf.has_parallel_hierarchy()) {
    *parallel_hierarchy = Shape(blob_conf.parallel_hierarchy());
  } else {
    *parallel_hierarchy = Shape({parallel_desc.parallel_num()});
  }
  LOG(INFO) << "input op InferParallelHierarchy" << parallel_hierarchy->DebugStr();
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferParallelDistributionSignature(
    ParallelDistributionSignature* signature, const SbpSignature& sbp_sig_conf,
    const ParallelDesc& parallel_desc, const Shape& parallel_hierarchy,
    std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
        ParallelDistributionInferHint4Ibn,
    std::function<Maybe<const OptInt64*>(const std::string&)> BatchAxis4BnInOp) {
  const InterfaceBlobConf& blob_conf = op_conf().input_conf().blob_conf();
  LOG(INFO) << "InputOp blob_conf" << blob_conf.DebugString();
  ParallelDistribution& in_parallel_distribution =
      (*signature->mutable_bn_in_op2parallel_distribution())["tick"];
  ParallelDistribution& out_parallel_distribution =
      (*signature->mutable_bn_in_op2parallel_distribution())["out"];
  if (blob_conf.has_parallel_distribution()) {
    out_parallel_distribution = blob_conf.parallel_distribution();
  } else {
    FOR_RANGE(int64_t, i, 0, parallel_hierarchy.NumAxes()) {
      out_parallel_distribution.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
    }
  }
  FOR_RANGE(int64_t, i, 0, parallel_hierarchy.NumAxes()) {
    in_parallel_distribution.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  }
  LOG(INFO) << "input op InferParallelDistributionSignature in:\n"
            << in_parallel_distribution.DebugString() << "\nout:\n"
            << out_parallel_distribution.DebugString();
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> InputOp::GetOpConfWithoutOpNameAndLbn() const {
  return SymbolOf(this->op_conf());
}

REGISTER_OP(OperatorConf::kInputConf, InputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kInputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kInputConf);

}  // namespace oneflow
