import unittest
import torch
import numpy as np
from sklearn.metrics import classification_report

from utils import get_test_subject_ids, fix_run_seed, calc_metrics



class UtilsTest(unittest.TestCase):
    def test_get_test_subject_ids(self):
        test_subjects = "TM_191008_01,TM_191009_01"
        test_subject_ids = get_test_subject_ids(test_subjects)
        
        self.assertEqual(len(test_subject_ids), 2)
        self.assertEqual(test_subject_ids[0], "TM_191008_01")
        self.assertEqual(test_subject_ids[1], "TM_191009_01")

        test_subjects = "TM_191008_01"
        test_subject_ids = get_test_subject_ids(test_subjects)
        
        self.assertEqual(len(test_subject_ids), 1)
        self.assertEqual(test_subject_ids[0], "TM_191008_01")

        test_subjects = ""
        test_subject_ids = get_test_subject_ids(test_subjects)
        
        self.assertEqual(len(test_subject_ids), 0)

        
    def test_fix_run_seed(self):
        fix_run_seed(0)
        a0 = torch.rand(1)
        
        fix_run_seed(0)
        a1 = torch.rand(1)
        
        self.assertEqual(a0, a1)

        
    def test_calc_metrics(self):
        recorded_labels = [0,     1,   0,   1,   0, 0, 1, 1]
        recorded_preds  = [0.1, 0.2, 0.5, 0.7, 0.8, 0.1, 0.6, 0.2]

        metrics = calc_metrics(recorded_labels, recorded_preds)
        print(metrics)


        binarised_preds = (np.array(recorded_preds) > 0.5)
        print(classification_report(recorded_labels, binarised_preds))
        
        
    
if __name__ == '__main__':
    unittest.main()
