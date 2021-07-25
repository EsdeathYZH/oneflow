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

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/autograd/autograd_meta.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

namespace one {

TensorInfo::TensorInfo(const Tensor& tensor)
    : shape_(tensor.shape()),
      dtype_(tensor.dtype()),
      device_(tensor.device()),
      parallel_desc_(tensor.parallel_desc()),
      parallel_distribution_(tensor.parallel_distribution()) {}

Maybe<const std::vector<int64_t>&> GetSbpTuple(
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  static thread_local HashMap<Symbol<cfg::ParallelDistribution>, std::vector<int64_t>> map;
  auto iter = map.find(parallel_distribution);
  if (iter == map.end()) {
    std::vector<int64_t> sbp_tuple;
    for (const auto& sbp_parallel : parallel_distribution->sbp_parallel()) {
      const auto& sbp_symbol = SymbolOf(sbp_parallel);
      sbp_tuple.push_back(*reinterpret_cast<const int64_t*>(&sbp_symbol));
    }
    iter = map.emplace(parallel_distribution, sbp_tuple).first;
  }
  return iter->second;
}

Maybe<Tensor> TensorInfo::zeros() const {
  if (TRY(device_).IsOk()) {
    const auto& device = JUST(device_);
    return functional::Constant(*shape_.get(), 0, dtype_,
                                *reinterpret_cast<const int64_t*>(&device));
  } else {
    const auto& parallel_desc = JUST(parallel_desc_);
    const auto& parallel_distribution = JUST(parallel_distribution_);
    const auto& sbp_tuple = JUST(GetSbpTuple(parallel_distribution));
    return functional::ConsistentConstant(
        *shape_.get(), 0, dtype_, *reinterpret_cast<const int64_t*>(&parallel_desc), sbp_tuple);
  }
}

}  // namespace one

}  // namespace oneflow
