import unittest
import torch

from utils import get_test_subject_ids, fix_run_seed



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
        
        
    
if __name__ == '__main__':
    unittest.main()
