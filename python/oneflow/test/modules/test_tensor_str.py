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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_tensor_str(test_case, device):
    # int dtype
    tensor = flow.tensor([[1, 2, 3], [4, 5, 6]], device=flow.device(device))
    tensor_str = str(tensor)
    test_case.assertTrue('5' in tensor_str)

    # empty
    tensor = flow.tensor([], device=flow.device(device))
    tensor_str = str(tensor)
    test_case.assertTrue('[]' in tensor_str)

    # scientific representation int_mode(val == np.ceil(val))
    tensor = flow.tensor([[1, 2, 3], [4, 5, 600000]], device=flow.device(device), dtype=flow.float64)
    tensor_str = str(tensor)
    test_case.assertTrue('6.0000e+05' in tensor_str)

    # int_mode
    tensor = flow.tensor([[1.0, 2.0, 3.0], [4.0, 5, 60]], device=flow.device(device), dtype=flow.float64)
    tensor_str = str(tensor)
    test_case.assertTrue('60.' in tensor_str)

    # float dtype
    tensor = flow.tensor([[1.3, 2.4, 3.5], [-4.6, 5, 60]], device=flow.device(device), dtype=flow.float64)
    tensor_str = str(tensor)
    test_case.assertTrue('60.0000' in tensor_str)

    # scientific representation float dtype
    tensor = flow.tensor([[1.3, 2.4, 3.5], [-4.6, 5, 60000000]], device=flow.device(device), dtype=flow.float64)
    tensor_str = str(tensor)
    test_case.assertTrue('3.5000e+00' in tensor_str)

    # summarized data float dtype
    tensor = flow.tensor(np.ones((100, 100, 100)), device=flow.device(device), dtype=flow.float64)
    tensor_str = str(tensor)
    test_case.assertTrue('...' in tensor_str)

@flow.unittest.skip_unless_1n1d()
class TestTensorStrModule(flow.unittest.TestCase):
    def test_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_tensor_str,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
