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

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_reshape_exception_only_one_dim_infered(test_case):
        # torch exception and messge:
        #
        #   RuntimeError: only one dimension can be inferred
        #
        x = flow.tensor((2, 2))
        with test_case.assertRaises(RuntimeError) as ctx:
            y = x.reshape((-1, -1))
        test_case.assertEqual("only one dimension can be inferred", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
