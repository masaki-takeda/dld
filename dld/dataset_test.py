import sys
import numpy as np
import unittest

import torch
from torch.utils.data import DataLoader

from dataset import BrainDataset, FACE_OBJECT
from dataset import DATA_TYPE_TRAIN, DATA_TYPE_VALIDATION, DATA_TYPE_TEST


class BrainDatasetTest(unittest.TestCase):

    def test_dataset(self):
        train_dataset = BrainDataset(data_type=DATA_TYPE_TRAIN,
                                     classify_type=FACE_OBJECT,
                                     data_seed=0,
                                     use_fmri=True,
                                     use_eeg=True,
                                     data_dir="./data2/DLD/Data_Converted",
                                     fmri_frame_type="normal",
                                     eeg_normalize_type="normal",
                                     eeg_frame_type="normal",
                                     use_smooth=True,
                                     fold=0,
                                     test_subjects=['TM_191008_01'],
                                     subjects_per_fold=1,
                                     debug=True)
        print(len(train_dataset.indices))
        #self.assertEqual(len(train_dataset.indices), 553)
        
        validation_dataset = BrainDataset(data_type=DATA_TYPE_VALIDATION,
                                          classify_type=FACE_OBJECT,
                                          data_seed=0,
                                          use_fmri=True,
                                          use_eeg=True,
                                          data_dir="./data2/DLD/Data_Converted",
                                          fmri_frame_type="normal",
                                          eeg_normalize_type="normal",
                                          eeg_frame_type="normal",
                                          use_smooth=True,
                                          fold=0,
                                          test_subjects=['TM_191008_01'],
                                          subjects_per_fold=1,
                                          debug=True)
        print(len(validation_dataset.indices))
        #self.assertEqual(len(validation_dataset.indices), 60)

        test_dataset = BrainDataset(data_type=DATA_TYPE_TEST,
                                    classify_type=FACE_OBJECT,
                                    data_seed=0,
                                    use_fmri=True,
                                    use_eeg=True,
                                    data_dir="./data2/DLD/Data_Converted",
                                    fmri_frame_type="normal",
                                    eeg_normalize_type="normal",
                                    eeg_frame_type="normal",
                                    use_smooth=True,
                                    fold=0,
                                    test_subjects=['TM_191008_01'],
                                    subjects_per_fold=1,
                                    debug=True)
        print(len(test_dataset.indices))
        #self.assertEqual(len(test_dataset.indices), 178)

        total_indices_size = len(train_dataset.indices) + len(validation_dataset.indices) + \
            len(test_dataset.indices)
        print(total_indices_size)

        #self.assertEqual(len(train_dataset.labels), total_indices_size)
        #self.assertEqual(len(train_dataset.eeg_datas), total_indices_size)

    """
    def test_dataset_ft(self):
        train_dataset = BrainDataset(data_type=DATA_TYPE_TRAIN,
                                     classify_type=FACE_OBJECT,
                                     data_seed=0,
                                     use_fmri=True,
                                     use_eeg=True,
                                     data_dir="./data2/DLD/Data_Converted",
                                     fmri_frame_type="normal",
                                     eeg_normalize_type="normal",
                                     eeg_frame_type="ft",
                                     use_smooth=True,
                                     fold=0,
                                     test_subjects=['TM_191008_01'],
                                     debug=True)
        self.assertEqual(len(train_dataset.indices), 553)
        
        validation_dataset = BrainDataset(data_type=DATA_TYPE_VALIDATION,
                                     classify_type=FACE_OBJECT,
                                     data_seed=0,
                                     use_fmri=True,
                                     use_eeg=True,
                                     data_dir="./data2/DLD/Data_Converted",
                                     fmri_frame_type="normal",
                                     eeg_normalize_type="normal",
                                     eeg_frame_type="ft",
                                     use_smooth=True,
                                     fold=0,
                                     test_subjects=['TM_191008_01'],
                                     debug=True)
        self.assertEqual(len(validation_dataset.indices), 60)


        test_dataset = BrainDataset(data_type=DATA_TYPE_TEST,
                                     classify_type=FACE_OBJECT,
                                     data_seed=0,
                                     use_fmri=True,
                                     use_eeg=True,
                                     data_dir="./data2/DLD/Data_Converted",
                                     fmri_frame_type="normal",
                                     eeg_normalize_type="normal",
                                     eeg_frame_type="ft",
                                     use_smooth=True,
                                     fold=0,
                                     test_subjects=['TM_191008_01'],
                                     debug=True)

        self.assertEqual(len(test_dataset.indices), 178)

        total_indices_size = len(train_dataset.indices) + len(validation_dataset.indices) + \
            len(test_dataset.indices)

        self.assertEqual(len(train_dataset.labels), total_indices_size)
        self.assertEqual(len(train_dataset.eeg_datas), total_indices_size)
    

    def test_dataset_average(self):
        train_dataset = BrainDataset(data_type=DATA_TYPE_TRAIN,
                                     classify_type=FACE_OBJECT,
                                     data_seed=0,
                                     use_fmri=True,
                                     use_eeg=True,
                                     data_dir="./data2/DLD/Data_Converted",
                                     fmri_frame_type="normal",
                                     eeg_normalize_type="normal",
                                     eeg_frame_type="normal",
                                     use_smooth=True,
                                     average_trial_size=3,
                                     average_repeat_size=4,
                                     fold=0,
                                     test_subjects=['TM_191008_01'],
                                     debug=True)
        self.assertEqual(len(train_dataset.indices), 735)
        
        validation_dataset = BrainDataset(data_type=DATA_TYPE_VALIDATION,
                                          classify_type=FACE_OBJECT,
                                          data_seed=0,
                                          use_fmri=True,
                                          use_eeg=True,
                                          data_dir="./data2/DLD/Data_Converted",
                                          fmri_frame_type="normal",
                                          eeg_normalize_type="normal",
                                          eeg_frame_type="normal",
                                          use_smooth=True,
                                          average_trial_size=3,
                                          average_repeat_size=4,
                                          fold=0,
                                          test_subjects=['TM_191008_01'],
                                          debug=True)
        self.assertEqual(len(validation_dataset.indices), 81)

        test_dataset = BrainDataset(data_type=DATA_TYPE_TEST,
                                    classify_type=FACE_OBJECT,
                                    data_seed=0,
                                    use_fmri=True,
                                    use_eeg=True,
                                    data_dir="./data2/DLD/Data_Converted",
                                    fmri_frame_type="normal",
                                    eeg_normalize_type="normal",
                                    eeg_frame_type="normal",
                                    average_trial_size=3,
                                    average_repeat_size=4,
                                    use_smooth=True,
                                    fold=0,
                                    test_subjects=['TM_191008_01'],
                                    debug=True)

        self.assertEqual(len(test_dataset.indices), 59)

        #total_indices_size = len(train_dataset.indices) + len(validation_dataset.indices) + \
        #    len(test_dataset.indices)
        #self.assertEqual(len(train_dataset.labels), total_indices_size)
        #self.assertEqual(len(train_dataset.eeg_datas), total_indices_size)
    """
        
    """
    def test_loader(self):
        # Datasets
        dataset = BrainDataset(data_type=DATA_TYPE_TRAIN,
                               classify_type=FACE_OBJECT,
                               data_seed=0,
                               use_fmri=True,
                               use_eeg=True,
                               data_dir="./data2/DLD/Data_Converted",
                               fmri_frame_type="normal",
                               eeg_normalize_type="normal",
                               eeg_frame_type="filter",
                               use_smooth=True,
                               fold=0,
                               subjects_per_fold=1,
                               debug=True)

        batch_size = 10
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

        for batch_idx, sample_batched in enumerate(loader):
            eeg_data = sample_batched['eeg_data']
            self.assertEqual(eeg_data.dtype, torch.float32)
            
            shape = eeg_data.numpy().shape
            self.assertEqual(len(shape), 4)
            self.assertLessEqual(shape[0], batch_size)
            self.assertEqual(shape[1], 5)
            self.assertEqual(shape[2], 63)
            self.assertEqual(shape[3], 250)

            fmri_data = sample_batched['fmri_data']
            self.assertEqual(fmri_data.dtype, torch.float32)

            shape = fmri_data.numpy().shape
            self.assertEqual(len(shape), 5)
            self.assertLessEqual(shape[0], batch_size)
            self.assertEqual(shape[1], 1)
            self.assertEqual(shape[2], 79)
            self.assertEqual(shape[3], 95)
            self.assertEqual(shape[4], 79)

            label_data = sample_batched['label']
            self.assertEqual(label_data.dtype, torch.float32)
            shape = label_data.numpy().shape
            self.assertLessEqual(len(shape), 2)
            self.assertLessEqual(shape[0], batch_size)
            self.assertEqual(shape[1], 1)

            break
    """


    # 以下のテストは未更新
    """
    def test_face_object(self):
        train_dataset = BrainDataset(train=True,
                                     classify_type=FACE_OBJECT,
                                     data_seed=0)
        test_dataset = BrainDataset(train=False,
                                    classify_type=FACE_OBJECT,
                                    data_seed=0)

        face_index  = 0
        object_index = 1

        self.assertTrue(train_dataset.labels[face_index] == 0 or
                        test_dataset.labels[face_index] == 0)
        self.assertTrue(train_dataset.labels[object_index] == 1 or
                        test_dataset.labels[object_index] == 1)

        self.assertTrue(face_index in train_dataset.indices or
                        face_index in test_dataset.indices)
        self.assertTrue(object_index in train_dataset.indices or
                        object_index in test_dataset.indices)
    """

    """
    def test_male_female(self):
        train_dataset = BrainDataset(train=True,
                                     classify_type=MALE_FEMALE,
                                     data_seed=0)
        test_dataset = BrainDataset(train=False,
                                    classify_type=MALE_FEMALE,
                                    data_seed=0)

        male_index   = 0
        female_index = 10

        self.assertTrue(train_dataset.labels[male_index] == 0 or
                        test_dataset.labels[male_index] == 0)
        self.assertTrue(train_dataset.labels[female_index] == 1 or
                        test_dataset.labels[female_index] == 1)

        self.assertTrue(male_index in train_dataset.indices or
                        male_index in test_dataset.indices)
        self.assertTrue(female_index in train_dataset.indices or
                        female_index in test_dataset.indices)

    def test_indoor_outdoor(self):
        train_dataset = BrainDataset(train=True,
                                     classify_type=INDOOR_OUTDOOR,
                                     data_seed=0)
        test_dataset = BrainDataset(train=False,
                                    classify_type=INDOOR_OUTDOOR,
                                    data_seed=0)

        indoor_index  = 20
        outdoor_index = 30

        self.assertTrue(train_dataset.labels[indoor_index] == 0 or
                        test_dataset.labels[indoor_index] == 0)
        self.assertTrue(train_dataset.labels[outdoor_index] == 1 or
                        test_dataset.labels[outdoor_index] == 1)

        self.assertTrue(indoor_index in train_dataset.indices or
                        indoor_index in test_dataset.indices)
        self.assertTrue(outdoor_index in train_dataset.indices or
                        outdoor_index in test_dataset.indices)
    """

    """
    def test_fold(self):
        fold = 9
        train_dataset = BrainDataset(train=True,
                                     classify_type=FACE_OBJECT,
                                     data_seed=0,
                                     fold=fold)
        test_dataset = BrainDataset(train=False,
                                    classify_type=FACE_OBJECT,
                                    data_seed=0,
                                    fold=fold)
        face_index  = 0
        object_index = 20

        self.assertTrue(train_dataset.labels[face_index] == 0 or
                        test_dataset.labels[face_index] == 0)
        self.assertTrue(train_dataset.labels[object_index] == 1 or
                        test_dataset.labels[object_index] == 1)

        self.assertTrue(face_index in train_dataset.indices or
                        face_index in test_dataset.indices)
        self.assertTrue(object_index in train_dataset.indices or
                        object_index in test_dataset.indices)

        self.assertTrue(len(test_dataset) < len(train_dataset))
        self.assertEqual(len(test_dataset)+len(train_dataset), 2277)
    """

    """
    def test_fmri(self):
        subject_id = 0
        dataset = BrainDataset(train=True,
                               classify_type=FACE_OBJECT,
                               data_seed=0,
                               use_fmri=True,
                               subject_id=subject_id)
        sample = dataset[0]
        fmri_data = sample['fmri_data']
        self.assertEqual(fmri_data.shape, (1, 79, 95, 79))
    """
        
        
if __name__ == '__main__':
    unittest.main()
