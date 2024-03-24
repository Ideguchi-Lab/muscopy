from __future__ import annotations

import os
import sys
import unittest

sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from muscopy.odt import find_max_args

seed = 42


class TestODT(unittest.TestCase):
    def test_find_max_args(self):
        np.random.seed(seed)
        size = 100
        array = np.zeros((size, size))
        random_point = np.random.randint(0, size, 2)
        array[random_point[0], random_point[1]] = 1
        max_x, max_y, value = find_max_args(array)
        assert max_x == random_point[0]
        assert max_y == random_point[1]


if __name__ == "__main__":
    unittest.main()
