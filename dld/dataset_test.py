import sys
import numpy as np
import unittest

import torch
from torch.utils.data import DataLoader

from dataset import BrainDataset, GROUP1_GROUP2
from dataset import CATEGORY_FACE, CATEGORY_OBJECT

class BrainDatasetTest(unittest.TestCase):

    def test_dataset(self):
        pass

    def test_group1_group2_indices(self):
        categories = np.array([
            CATEGORY_OBJECT,
            CATEGORY_OBJECT,
            CATEGORY_OBJECT,
            CATEGORY_OBJECT,
            CATEGORY_FACE,
            CATEGORY_OBJECT,
            CATEGORY_OBJECT,
            CATEGORY_OBJECT,
            CATEGORY_OBJECT,
            CATEGORY_OBJECT,
        ], dtype=np.int32)
        sub_categories = np.ones(len(categories), dtype=np.int32) * -1
        identities = np.array([0, 1, 2, 3, 0, -4, -5, 3, 2, 0], dtype=np.int32)
        angles = np.ones(len(categories), dtype=np.int32) * -1
        trial_mask = np.array([True, True, True, True, True, True, True, False, False, True])

        indices0, indices1 = BrainDataset.get_indices(
            None,
            GROUP1_GROUP2,
            categories,
            sub_categories,
            identities,
            angles,
            trial_mask,
            for_test=False)

        np.testing.assert_array_equal(indices0, np.array([0, 3, 5, 9]))
        np.testing.assert_array_equal(indices1, np.array([1, 2, 6]))
        
        
if __name__ == '__main__':
    unittest.main()
