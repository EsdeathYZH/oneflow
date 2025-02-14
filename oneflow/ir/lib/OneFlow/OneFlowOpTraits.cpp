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
#include "OneFlow/OneFlowOps.h"

namespace mlir {

namespace OpTrait {

namespace {

// TODO: merge all ctrl input and output when folding op
bool HaveIdenticalPlacement(mlir::Operation* a, mlir::Operation* b) {
  oneflow::UserOpAdaptor adaptor_a(a->getOperands(), a->getAttrDictionary());
  oneflow::UserOpAdaptor adaptor_b(b->getOperands(), b->getAttrDictionary());
  return adaptor_a.device_tag() == adaptor_b.device_tag()
         && adaptor_a.device_name() == adaptor_b.device_name();
}

}  // namespace

namespace impl {

OpFoldResult foldIdempotentOfIdenticalPlacement(Operation* op) {
  auto* argument_op = op->getOperand(0).getDefiningOp();
  if (argument_op && op->getName() == argument_op->getName()
      && HaveIdenticalPlacement(op, argument_op)) {
    return op->getOperand(0);
  }
  return {};
}

OpFoldResult foldInvolutionOfIdenticalPlacement(Operation* op) {
  auto* argument_op = op->getOperand(0).getDefiningOp();
  if (argument_op && op->getName() == argument_op->getName()
      && HaveIdenticalPlacement(op, argument_op)) {
    return argument_op->getOperand(0);
  }
  return {};
}

LogicalResult VerifyIsOpConfCompatible(Operation* op) {
  for (auto attr : {
           IsOpConfCompatible<void>::getOpNameAttr(),
           IsOpConfCompatible<void>::getDeviceTagAttr(),
       }) {
    if (!op->hasAttrOfType<StringAttr>(attr)) {
      return op->emitError("expected operation to have attribute: " + attr);
    }
  }
  if (!op->hasAttrOfType<ArrayAttr>(IsOpConfCompatible<void>::getDeviceNameAttr())) {
    return op->emitError("expected operation to have attribute: "
                         + IsOpConfCompatible<void>::getDeviceNameAttr());
  }
  return success();
}

LogicalResult VerifyIsImportCompatible(Operation* op) {
  if (auto output_lbns =
          op->getAttrOfType<ArrayAttr>(IsImportCompatible<void>::getOutputLBNsAttr())) {
    if (auto cec = dyn_cast<oneflow::ControlEdgeCompatible>(op)) {
      if (cec.dataOutputResults().size() != output_lbns.size()) {
        return op->emitError("expected number of data output results to be "
                             + std::to_string(output_lbns.size()) + " but got "
                             + std::to_string(cec.dataOutputResults().size()));
      }
    } else {
      return op->emitError("expected to support ControlEdgeCompatible");
    }
  } else {
    return op->emitError("expected operation to have attribute: "
                         + IsImportCompatible<void>::getOutputLBNsAttr());
  }
  return success();
}

}  // namespace impl

}  // namespace OpTrait

}  // namespace mlir
