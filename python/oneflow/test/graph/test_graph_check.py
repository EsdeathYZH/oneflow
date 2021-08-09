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
import os
import unittest
import sys

import numpy as np

import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphCheck(flow.unittest.TestCase):
    def test_graph_input_output(test_case):
        class CustomModuleIOCheck(flow.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        class CustomGraphIOCheck(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleIOCheck()

            def build(self, x):
                return self.m(x)

        g = CustomGraphIOCheck()
        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        out = g(x)
        test_case.assertTrue(np.array_equal(x.numpy(), out.numpy()))

    def test_tensor_numpy_check(test_case):
        class CustomModuleNumpyCheck(flow.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.numpy()
                return x

        class CustomGraphNumpyCheck(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.m = CustomModuleNumpyCheck()

            def build(self, x):
                return self.m(x)

        g = CustomGraphNumpyCheck()
        x = np.ones((10, 10))
        x = flow.tensor(x, dtype=flow.float32)
        try:
            out = g(x)
        except:
            print(sys.exc_info())


if __name__ == "__main__":
    unittest.main()
