import sys
import numpy as np
import unittest

from behavior import Behavior
from fmri import FMRI


class FMRITest(unittest.TestCase):
    def test_init(self):
        src_base = "/Users/miyoshi/Desktop/kut_data/Data"
        valid   = True

        """
        date    = 191008
        subject = 1
        run     = 1
        reject_trials = [17, 22, 30, 33]
        data_shape = (37, 79, 95, 79)
        """
        
        date    = 191016
        subject = 1
        run     = 2
        reject_trials = [27,38,40,41,43,45,46,48,50]
        data_shape = (31, 79, 95, 79)
        
        behavior = Behavior(src_base, date, subject, run, reject_trials)

        # For normal
        fmri0 = FMRI(src_base=src_base,
                    behavior=behavior,
                    normalize_per_run=True,
                    use_frame_average=False)
        
        self.assertEqual(fmri0.data.shape, data_shape)
        self.assertEqual(fmri0.data.dtype, np.float32)

        # For using the average data of 3TR
        fmri1 = FMRI(src_base=src_base,
                     behavior=behavior,
                     normalize_per_run=True,
                     use_frame_average=True)
        self.assertEqual(fmri1.data.shape, data_shape)
        self.assertEqual(fmri1.data.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
