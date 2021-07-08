"""
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
"""
import unittest
from collections import OrderedDict

import numpy as np

from oneflow.compatible.single_client import experimental as flow
from test_util import GenArgList


def _test_ceil_impl(test_case, device, shape):
    x = flow.Tensor(
        np.random.randn(*shape), device=flow.device(device), requires_grad=True
    )
    of_out = flow.ceil(x)
    np_out = np.ceil(x.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), np.zeros(shape), 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCeilModule(flow.unittest.TestCase):
    def test_ceil(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_ceil_impl]

        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(1,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
