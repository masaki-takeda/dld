import numpy as np
import unittest

from preprocess_average import Subject, AveragingBehavior, preprocess_average_behavior
from dataset import FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, GROUP1_GROUP2
from dataset import CATEGORY_FACE, CATEGORY_OBJECT, SUBCATEGORY_MALE, SUBCATEGORY_FEMALE, SUBCATEGORY_ARTIFICIAL, SUBCATEGORY_NATURAL


class PreprocessAverageTest(unittest.TestCase):
    def test_subject(self):
        np.random.seed(0)
        
        indices0 = [0, 1, 2, 3]
        indices1 = [10, 11, 12, 13, 14]

        subject_id = "TM0000"
        
        subject_obj = Subject(subject_id,
                              indices0,
                              indices1,
                              average_trial_size=3,
                              average_repeat_size=4)

        self.assertEqual(subject_obj.averaging_indices0.shape, (5, 3)) # (4*4) // 3 = 5
        self.assertEqual(subject_obj.averaging_indices1.shape, (6, 3)) # (5*4) // 3 = 6

        np.testing.assert_array_equal(subject_obj.averaging_repeat_indices0,
                                      np.array([0, 1, 2, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(subject_obj.averaging_repeat_indices1,
                                      np.array([0, 1, 1, 2, 2, 3], dtype=np.int32))

        self.assertEqual(len(subject_obj.subject_ids0), 5)
        self.assertEqual(len(subject_obj.subject_ids1), 6)

        subject_obj.process_unmatched()

        print(subject_obj.averaging_indices0)
        print(subject_obj.alt_averaging_indices0)

        print(subject_obj.averaging_indices1)
        print(subject_obj.alt_averaging_indices1)

        
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
                                                   alt_indices0=None,
                                                   alt_indices1=None,
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
                                                   alt_indices0=None,
                                                   alt_indices1=None,
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
                                                   alt_indices0=None,
                                                   alt_indices1=None,
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

        # CT5
        averaging_behavior_ct5 = AveragingBehavior(classify_type=GROUP1_GROUP2,
                                                   indices0=indices0,
                                                   indices1=indices1,
                                                   alt_indices0=None,
                                                   alt_indices1=None,
                                                   repeat_indices0=repeat_indices0,
                                                   repeat_indices1=repeat_indices1,
                                                   subject_ids0=subject_ids0,
                                                   subject_ids1=subject_ids1)

        self.assertEqual(averaging_behavior_ct5.indices.shape, (5, 3))
        self.assertEqual(averaging_behavior_ct5.repeat_indices.shape, (5,))

        np.testing.assert_array_equal(averaging_behavior_ct5.categories,
                                      np.array([CATEGORY_OBJECT, CATEGORY_OBJECT, CATEGORY_OBJECT,
                                                CATEGORY_OBJECT, CATEGORY_OBJECT],
                                               dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior_ct5.sub_categories,
                                      np.array([-1, -1, -1, -1, -1],
                                               dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior_ct5.identities,
                                      np.array([-4, -4, -4, -5, -5],
                                               dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior_ct5.subject_ids,
                                      ['TM0000', 'TM0000', 'TM0001','TM0000', 'TM0001'])

    def test_preprocess_average_behavior_group1_group2(self):
        np.random.seed(0)
        behavior_data = {
            "category": np.array([
                CATEGORY_OBJECT,
                CATEGORY_OBJECT,
                CATEGORY_OBJECT,
                CATEGORY_OBJECT,
                CATEGORY_FACE,
                CATEGORY_OBJECT,
                CATEGORY_OBJECT,
                CATEGORY_OBJECT,
                CATEGORY_OBJECT,
            ], dtype=np.int32),
            "sub_category": np.ones(9, dtype=np.int32) * -1,
            "subject": np.array([
                "TM0000",
                "TM0000",
                "TM0000",
                "TM0000",
                "TM0000",
                "TM0001",
                "TM0001",
                "TM0001",
                "TM0001",
            ]),
            "identity": np.array([0, 1, 2, 3, 0, 0, 1, 2, 3], dtype=np.int32),
            "angle": np.ones(9, dtype=np.int32) * -1,
        }

        averaging_behavior = preprocess_average_behavior(
            behavior_data,
            classify_type=GROUP1_GROUP2,
            average_trial_size=1,
            average_repeat_size=1,
            unmatched=False)

        np.testing.assert_array_equal(np.sort(averaging_behavior.indices0.reshape(-1)),
                                      np.array([0, 3, 5, 8]))
        np.testing.assert_array_equal(np.sort(averaging_behavior.indices1.reshape(-1)),
                                      np.array([1, 2, 6, 7]))
        np.testing.assert_array_equal(averaging_behavior.categories,
                                      np.array([CATEGORY_OBJECT] * 8, dtype=np.int32))
        np.testing.assert_array_equal(averaging_behavior.identities,
                                      np.array([-4, -4, -4, -4, -5, -5, -5, -5],
                                               dtype=np.int32))
        
        
if __name__ == '__main__':
    unittest.main()
