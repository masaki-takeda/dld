import numpy as np
import unittest

from preprocess_average import Subject, AveragingBehavior
from dataset import FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL
from dataset import CATEGORY_FACE, CATEGORY_OBJECT, SUBCATEGORY_MALE, SUBCATEGORY_FEMALE, SUBCATEGORY_ARTIFICIAL, SUBCATEGORY_NATURAL


class PreprocessAverageTest(unittest.TestCase):
    def test_subject(self):
        indices0 = [0, 1, 2, 3]
        indices1 = [10, 11, 12, 13, 14]

        subject_id = "TM0000"
        
        subject_obj = Subject(subject_id,
                              indices0,
                              indices1,
                              average_trial_size=3,
                              average_repeat_size=4)

        self.assertEqual(subject_obj.averaging_indices0.shape, (5, 3)) # (4*4) // 3
        self.assertEqual(subject_obj.averaging_indices1.shape, (6, 3)) # (5*4) // 3

        np.testing.assert_array_equal(subject_obj.averaging_repeat_indices0,
                                      np.array([0, 1, 2, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(subject_obj.averaging_repeat_indices1,
                                      np.array([0, 1, 1, 2, 2, 3], dtype=np.int32))

        self.assertEqual(len(subject_obj.subject_ids0), 5)
        self.assertEqual(len(subject_obj.subject_ids1), 6)

        
    def test_averaging_behavior(self):
        indices0 = np.array([[0,1,2],[1,2,3],[2,3,4]], dtype=np.int32)
        indices1 = np.array([[10,11,12],[11,12,13]], dtype=np.int32)

        subject_ids0 = ['TM0000', 'TM0000', 'TM0001']
        subject_ids1 = ['TM0000', 'TM0001']
        
        repeat_indices0 = np.array([0,0,0], dtype=np.int32)
        repeat_indices1 = np.array([1,2], dtype=np.int32)

        # CT0
        averaging_behavior_ct0 = AveragingBehavior(classify_type=FACE_OBJECT,
                                                   indices0=indices0,
                                                   indices1=indices1,
                                                   repeat_indices0=repeat_indices0,
                                                   repeat_indices1=repeat_indices1,
                                                   subject_ids0=subject_ids0,
                                                   subject_ids1=subject_ids1)

        self.assertEqual(averaging_behavior_ct0.indices.shape, (5, 3))
        self.assertEqual(averaging_behavior_ct0.repeat_indices.shape, (5,))

        np.testing.assert_array_equal(averaging_behavior_ct0.categories,
                                      np.array([FACE_OBJECT, FACE_OBJECT, FACE_OBJECT,
                                                CATEGORY_OBJECT, CATEGORY_OBJECT],
                                               dtype=np.int32))

        np.testing.assert_array_equal(averaging_behavior_ct0.sub_categories,
                                      np.array([-1,-1,-1,-1,-1],
                                               dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior_ct0.subject_ids,
                                      ['TM0000', 'TM0000', 'TM0001','TM0000', 'TM0001'])

        # CT1
        averaging_behavior_ct1 = AveragingBehavior(classify_type=MALE_FEMALE,
                                                   indices0=indices0,
                                                   indices1=indices1,
                                                   repeat_indices0=repeat_indices0,
                                                   repeat_indices1=repeat_indices1,
                                                   subject_ids0=subject_ids0,
                                                   subject_ids1=subject_ids1)
        
        self.assertEqual(averaging_behavior_ct1.indices.shape, (5, 3))
        self.assertEqual(averaging_behavior_ct1.repeat_indices.shape, (5,))

        np.testing.assert_array_equal(averaging_behavior_ct1.categories,
                                      np.array([CATEGORY_FACE, CATEGORY_FACE, CATEGORY_FACE,
                                                CATEGORY_FACE, CATEGORY_FACE],
                                               dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior_ct1.sub_categories,
                                      np.array([SUBCATEGORY_MALE, SUBCATEGORY_MALE, SUBCATEGORY_MALE,
                                                SUBCATEGORY_FEMALE, SUBCATEGORY_FEMALE],
                                               dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior_ct1.subject_ids,
                                      ['TM0000', 'TM0000', 'TM0001','TM0000', 'TM0001'])        
        
        # CT2
        averaging_behavior_ct2 = AveragingBehavior(classify_type=ARTIFICIAL_NATURAL,
                                                   indices0=indices0,
                                                   indices1=indices1,
                                                   repeat_indices0=repeat_indices0,
                                                   repeat_indices1=repeat_indices1,
                                                   subject_ids0=subject_ids0,
                                                   subject_ids1=subject_ids1)
        
        self.assertEqual(averaging_behavior_ct2.indices.shape, (5, 3))
        self.assertEqual(averaging_behavior_ct2.repeat_indices.shape, (5,))

        np.testing.assert_array_equal(averaging_behavior_ct2.categories,
                                      np.array([CATEGORY_OBJECT, CATEGORY_OBJECT, CATEGORY_OBJECT,
                                                CATEGORY_OBJECT, CATEGORY_OBJECT],
                                               dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior_ct2.sub_categories,
                                      np.array([SUBCATEGORY_ARTIFICIAL,
                                                SUBCATEGORY_ARTIFICIAL,
                                                SUBCATEGORY_ARTIFICIAL,
                                                SUBCATEGORY_NATURAL,
                                                SUBCATEGORY_NATURAL],
                                               dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior_ct2.subject_ids,
                                      ['TM0000', 'TM0000', 'TM0001','TM0000', 'TM0001'])
        
        
if __name__ == '__main__':
    unittest.main()
