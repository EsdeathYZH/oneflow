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
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/job/placement_scope.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace one {

class InputConsistentTensorMeta final {
 public:
  InputConsistentTensorMeta()
      : tensor_meta_(nullptr),
        consumer_forced_parallel_distribution_(consumer_forced_parallel_distribution) {}
  InputConsistentTensorMeta(Symbol<ConsistentTensorMeta> tensor_meta,
                            Symbol<cfg::ParallelDistribution> consumer_forced_parallel_distribution)
      : tensor_meta_(tensor_meta),
        consumer_forced_parallel_distribution_(consumer_forced_parallel_distribution) {}

  InputConsistentTensorMeta(const InputConsistentTensorMeta&) = default;
  InputConsistentTensorMeta(InputConsistentTensorMeta&&) = default;
  ~InputConsistentTensorMeta() = default;

  size_t hash_value() const {
    return std::hash<Symbol<ConsistentTensorMeta>>()(tensor_meta())
           ^ std::hash<Symbol<cfg::ParallelDistribution>>()(
               consumer_forced_parallel_distribution());
  }

  bool operator==(const InputConsistentTensorMeta& other) const {
    return this->tensor_meta() == other.tensor_meta()
           && this->consumer_forced_parallel_distribution()
                  == other.consumer_forced_parallel_distribution();
  }

  Symbol<ConsistentTensorMeta> tensor_meta() const { return tensor_meta_; }
  Symbol<cfg::ParallelDistribution> consumer_forced_parallel_distribution() const {
    return consumer_forced_parallel_distribution_;
  }
  void assign(Symbol<ConsistentTensorMeta> tensor_meta,
              Symbol<cfg::ParallelDistribution> consumer_forced_parallel_distribution) {
    tensor_meta_ = tensor_meta;
    consumer_forced_parallel_distribution_ = consumer_forced_parallel_distribution;
  }

 private:
  Symbol<ConsistentTensorMeta> tensor_meta_;
  Symbol<cfg::ParallelDistribution> consumer_forced_parallel_distribution_;
};

class ConsistentTensorMetaInferArgs final {
 public:
  ConsistentTensorMetaInferArgs(size_t input_tensor_size, Symbol<PlacementScope> placement_scope,
                                const AttrMap& attrs)
      : input_consistent_tensor_metas_(input_tensor_size),
        placement_scope_(placement_scope),
        attrs_(attrs) {}

  ConsistentTensorMetaInferArgs(const ConsistentTensorMetaInferArgs&) = default;
  ConsistentTensorMetaInferArgs(ConsistentTensorMetaInferArgs&&) = default;
  ~ConsistentTensorMetaInferArgs() = default;

  const std::vector<InputConsistentTensorMeta>& input_consistent_tensor_metas() const {
    return input_consistent_tensor_metas_;
  }
  Symbol<PlacementScope> placement_scope() const { return placement_scope_; }
  const AttrMap& attrs() const { return attrs_; }
  size_t hash_value() const {
    size_t hash_value = std::hash<Symbol<PlacementScope>>()(placement_scope_);
    hash_value ^= std::hash<AttrMap>()(attrs_);
    const auto& tensor_meta_hash_functor = std::hash<InputConsistentTensorMeta>();
    for (const auto& tensor_meta : input_consistent_tensor_metas_) {
      HashCombine(&hash_value, tensor_meta_hash_functor(tensor_meta));
    }
    return hash_value;
  }

  bool operator==(const ConsistentTensorMetaInferArgs& other) const {
    return this->input_consistent_tensor_metas_ == other.input_consistent_tensor_metas_
           && this->placement_scope_ == other.placement_scope_ && this->attrs_ == other.attrs_;
  }

  void InitInputConsistentTensorMetas(const TensorTuple& input_tensors) {
    CHECK_EQ(input_consistent_tensor_metas_.size(), input_tensors.size());
    for (const auto& tensor : input_tensors) {
      input_consistent_tensor_metas_.assign(tensor->tensor_meta(),
                                            tensor->consumer_forced_parallel_distribution());
    }
  }

 private:
  std::vector<InputConsistentTensorMeta> input_consistent_tensor_metas_;
  Symbol<PlacementScope> placement_scope_;
  AttrMap attrs_;
};

class ConsistentTensorMetaInferResult final {
 public:
 private:
  std::shared_ptr<TensorTuple> output_tensors_;
};

std::shared_ptr<ConsistentTensorMetaInferResult> Infer(
    const UserOpExpr& user_op_expr, const ConsistentTensorMetaInferArgs& infer_args) {
  TODO();
  return std::shared_ptr<ConsistentTensorMetaInferResult>();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::one::InputConsistentTensorMeta> final {
  size_t operator()(const oneflow::one::InputConsistentTensorMeta& val) const {
    return val.hash_value();
  }
};

template<>
struct hash<oneflow::one::ConsistentTensorMetaInferArgs> final {
  size_t operator()(const oneflow::one::ConsistentTensorMetaInferArgs& val) const {
    return val.hash_value();
  }
};

}  // namespace std
