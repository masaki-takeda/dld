import sys
import numpy as np
import unittest

from eeg import EEG


class EEGTest(unittest.TestCase):
    def test_init(self):
        # TDOO: 
        eeg = EEG(date="191008", subject=1, run=1)
        self.assertEqual(eeg.data.shape, (63, 375, 50))
        
        
if __name__ == '__main__':
    unittest.main()
