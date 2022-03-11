import numpy as np
import os
from torch.utils.data import Dataset

from eeg import EEG
from behavior import Behavior


# data_typeに与える値
DATA_TYPE_TRAIN      = 0
DATA_TYPE_VALIDATION = 1
DATA_TYPE_TEST       = 2

# Classification Type
FACE_OBJECT        = 0
MALE_FEMALE        = 1
ARTIFICIAL_NATURAL = 2
CLASSIFY_ALL       = -1

CATEGORY_FACE          = 0
CATEGORY_OBJECT        = 1
SUBCATEGORY_MALE       = 0
SUBCATEGORY_FEMALE     = 1
SUBCATEGORY_ARTIFICIAL = 0
SUBCATEGORY_NATURAL    = 1

FMRI_FRAME_TYPE_NORMAL  = 0
FMRI_FRAME_TYPE_AVERAGE = 1
FMRI_FRAME_TYPE_THREE   = 2

EEG_NORMALIZE_TYPE_NORMAL  = 0
EEG_NORMALIZE_TYPE_PRE     = 1
EEG_NORMALIZE_TYPE_NONE    = 2

EEG_FRAME_TYPE_NORMAL = 0
EEG_FRAME_TYPE_FILTER = 1
EEG_FRAME_TYPE_FT     = 2


# マスクを利用する場合Trueにする
DEBUG_USE_EEG_MASK = False

# マスクに指定するChannelの数値
CH_Fp1  = 0
CH_Fp2  = 1
CH_F3   = 2
CH_F4   = 3
CH_C3   = 4
CH_C4   = 5
CH_P3   = 6
CH_P4   = 7
CH_O1   = 8
CH_O2   = 9
CH_F7   = 10
CH_F8   = 11
CH_T7   = 12
CH_T8   = 13
CH_P7   = 14
CH_P8   = 15
CH_Fz   = 16
CH_Cz   = 17
CH_Pz   = 18
CH_Oz   = 19
CH_FC1  = 20
CH_FC2  = 21
CH_CP1  = 22
CH_CP2  = 23
CH_FC5  = 24
CH_FC6  = 25
CH_CP5  = 26
CH_CP6  = 27
CH_TP9  = 28
CH_TP10 = 29
CH_POz  = 30
CH_F1   = 31
CH_F2   = 32
CH_C1   = 33
CH_C2   = 34
CH_P1   = 35
CH_P2   = 36
CH_AF3  = 37
CH_AF4  = 38
CH_FC3  = 39
CH_FC4  = 40
CH_CP3  = 41
CH_CP4  = 42
CH_PO3  = 43
CH_PO4  = 44
CH_F5   = 45
CH_F6   = 46
CH_C5   = 47
CH_C6   = 48
CH_P5   = 49
CH_P6   = 50
CH_AF7  = 51
CH_AF8  = 52
CH_FT7  = 53
CH_FT8  = 54
CH_TP7  = 55
CH_TP8  = 56
CH_PO7  = 57
CH_PO8  = 58
CH_FT9  = 59
CH_FT10 = 60
CH_Fpz  = 61
CH_CPz  = 62


class BrainDataset(Dataset):
    def __init__(self,
                 data_type,
                 classify_type,
                 data_seed,
                 use_fmri=False,
                 use_eeg=False,
                 data_dir="./data",
                 fmri_frame_type="normal",
                 eeg_normalize_type="normal",
                 eeg_frame_type="normal",
                 use_smooth=True,
                 average_trial_size=0,
                 average_repeat_size=0,
                 fold=0,
                 test_subjects=[],
                 subjects_per_fold=4,
                 debug=False):
        """
        use_fmri:
           (bool) fMRIデータを利用するかどうか
        use_eeg:
           (bool) EEGデータを利用するかどうか
        fmri_frame_type:
           (str) fMRIデータで1TRのみ利用, 3TRの平均を利用 or 3TR全部を利用
        eeg_normalize_type:
           (str) EEGのノーマライズタイプ
        eeg_frame_type:
           (str) EEGデータのタイプ. (normal, filter, ft)
        use_smooth:
           (bool) fMRIでsmoothingを適用したデータを利用するかどうか
        average_trial_size:
           (int) 平均をとるtrial数
        average_repeat_size:
           (int) 平均をとる時のリピート数
        subjects_per_fold:
           (int) 1Foldあたりの被験者数
        """
        self.use_fmri = use_fmri
        self.use_eeg = use_eeg
        self.data_dir = data_dir
        self.use_smooth = use_smooth

        assert (data_type == DATA_TYPE_TRAIN or data_type == DATA_TYPE_VALIDATION or data_type == DATA_TYPE_TEST)
        assert (fmri_frame_type == "normal" or fmri_frame_type == "average" or fmri_frame_type == "three")
        assert (eeg_normalize_type == "normal" or eeg_normalize_type == "pre" or eeg_normalize_type == "none")
        assert (eeg_frame_type == "normal" or eeg_frame_type == "filter" or eeg_frame_type == "ft")

        print("fmri frame type={}, eeg normalize type={}, eeg frame type={}".format(
            fmri_frame_type, eeg_normalize_type, eeg_frame_type))
        
        if fmri_frame_type == "normal":
            self.fmri_frame_type = FMRI_FRAME_TYPE_NORMAL
        elif fmri_frame_type == "average":
            self.fmri_frame_type = FMRI_FRAME_TYPE_AVERAGE
        else:
            self.fmri_frame_type = FMRI_FRAME_TYPE_THREE

        if eeg_normalize_type == "normal":
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_NORMAL
        elif eeg_normalize_type == "pre":
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_PRE
        else:
            self.eeg_normalize_type = EEG_NORMALIZE_TYPE_NONE

        if eeg_frame_type == "normal":
            self.eeg_frame_type = EEG_FRAME_TYPE_NORMAL
        elif eeg_frame_type == "filter":
            self.eeg_frame_type = EEG_FRAME_TYPE_FILTER
        else:
            self.eeg_frame_type = EEG_FRAME_TYPE_FT

        if average_trial_size > 0:
            self.use_trial_average = True
        else:
            self.use_trial_average = False

        if self.use_eeg:
            # EEGデータロード
            eeg_suffix = ""
            if self.eeg_frame_type == EEG_FRAME_TYPE_FILTER:
                # EEGにてfilterを利用する場合はファイル名に"_filter"がつく
                eeg_suffix = "_filter"
            elif self.eeg_frame_type == EEG_FRAME_TYPE_FT:
                # EEGにてFT spectrogramを利用する場合はファイル名に"_ft"がつく
                eeg_suffix = "_ft"

            if self.eeg_normalize_type == EEG_NORMALIZE_TYPE_NORMAL:
                eeg_data_path = os.path.join(self.data_dir, "final_eeg_data{}".format(
                    eeg_suffix))
            elif self.eeg_normalize_type == EEG_NORMALIZE_TYPE_PRE:
                eeg_data_path = os.path.join(self.data_dir, "final_eeg_data_pre{}".format(
                    eeg_suffix))
            else:
                eeg_data_path = os.path.join(self.data_dir, "final_eeg_data_none{}".format(
                    eeg_suffix))

            if self.use_trial_average:
                # Trial平均利用時
                eeg_data_path = eeg_data_path + "_a{}_r{}_ct{}".format(average_trial_size,
                                                                       average_repeat_size,
                                                                       classify_type)
            eeg_data_path = eeg_data_path + ".npz"
            
            eeg_data_all = np.load(eeg_data_path)
            eeg_datas = eeg_data_all["eeg_data"] # (3940, 63, 375) or (3940, 5, 63, 375)等
            
            self.eeg_datas = eeg_datas

        if self.use_fmri:
            # fMRIのデータディレクトリ設定
            if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL:
                # 通常の場合
                fmri_data_dir = "final_fmri_data"
            elif self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
                # 3TRの平均を利用する場合
                fmri_data_dir = "final_fmri_data_av"
            else:
                # 3TR全部を利用する場合
                fmri_data_dir = "final_fmri_data_th"

            if not self.use_smooth:
                # smoothingをかけていない場合に_nsをお尻に付加したディレクトリ名となっている
                fmri_data_dir = fmri_data_dir + "_nosmooth"

            if self.use_trial_average:
                # Trial平均利用時
                fmri_data_dir = fmri_data_dir + "_a{}_r{}_ct{}".format(
                    average_trial_size,
                    average_repeat_size,
                    classify_type)
            
            self.fmri_data_dir = os.path.join(self.data_dir, fmri_data_dir)
            
        # データは全て持っておき、indexだけを変えるというやり方を取っている.
        
        # behaviorデータのロード
        behavior_data_path = os.path.join(self.data_dir, "final_behavior_data")

        if self.use_trial_average:
            behavior_data_path = behavior_data_path + "_a{}_r{}_ct{}".format(
                average_trial_size, average_repeat_size, classify_type)
        if debug:
            behavior_data_path = behavior_data_path + "_debug"
        
        behavior_data_all = np.load(behavior_data_path + ".npz")
        
        categories     = behavior_data_all["category"]     # (3940),
        sub_categories = behavior_data_all["sub_category"] # (3940),
        subjects       = behavior_data_all["subject"]      # (3940),

        #if self.use_trial_average:
        #    self.average_repeat_indices = behavior_data_all["repeat_index"]

        unique_subjects = np.unique(behavior_data_all['subject'])
        unique_subjects = np.sort(unique_subjects)

        train_validation_subjects = np.sort(list(set(unique_subjects) - set(test_subjects)))

        # ここで乱数を0で一旦固定 (被験者の分割はseedによらないようにする為)
        np.random.seed(0)
        # train&validation用の被験者IDをshuffleする
        np.random.shuffle(train_validation_subjects)
        
        # 引数のdata_seedで乱数を固定
        if data_seed != 0:
            # 以前のものと互換性を持たせるために、0の時は上の被験者seedでの固定を
            # 継続して利用するようにしている.
            np.random.seed(data_seed)

        if debug:
            # デバッグ時はfoldあたりの被験者数を強制的に1にする.
            subjects_per_fold = 1

        assert fold < len(train_validation_subjects) // subjects_per_fold

        # train/validationに被験者を分ける.
        # (被験者をsubjects_per_fold人ずつ分割して分ける)
        train_subjects = []
        validation_subjects = None

        for i in range(len(train_validation_subjects) // subjects_per_fold):
            fold_subjects = train_validation_subjects[i*subjects_per_fold:(i+1)*subjects_per_fold]
            if i == fold:
                validation_subjects = list(fold_subjects)
            else:
                train_subjects += list(fold_subjects)
        
        # test,train,validationのTrialの場所の[True, False, ....]
        test_subject_set       = set(test_subjects)
        train_subject_set      = set(train_subjects)
        validation_subject_set = set(validation_subjects)

        test_trials       = np.array([subject in test_subject_set       for subject in subjects])
        train_trials      = np.array([subject in train_subject_set      for subject in subjects])
        validation_trials = np.array([subject in validation_subject_set for subject in subjects])

        test_indices       = np.where(test_trials == True)[0]
        train_indices      = np.where(train_trials == True)[0]
        vaidation_indices  = np.where(validation_trials == True)[0]

        if data_type == DATA_TYPE_TRAIN:
            trial_mask = train_trials
        elif data_type == DATA_TYPE_VALIDATION:
            trial_mask = validation_trials
        elif data_type == DATA_TYPE_TEST:
            trial_mask = test_trials
        
        all_indices0, all_indices1 = self.get_indices(classify_type,
                                                      categories,
                                                      sub_categories,
                                                      trial_mask,
                                                      for_test=(data_type==DATA_TYPE_TEST))
                
        if data_type == DATA_TYPE_TRAIN:
            # Trainの場合はシャッフルする
            # ラベル0, ラベル1対象のindex配列(元データでの位置)シャッフルする
            np.random.shuffle(all_indices0)
            np.random.shuffle(all_indices1)
            
        indices0 = all_indices0 # Class0(Faceなど)のindex配列
        indices1 = all_indices1 # Class1(Objectなど)のindex配列

        if data_type == DATA_TYPE_TEST:
            # TESTデータはpositive,negativeの数を揃えておく.
            min_size = np.min([len(indices0), len(indices1)])
            indices0 = indices0[:min_size]
            indices1 = indices1[:min_size]
        
        # ラベル用のint arrayを用意し, class0の場所には0, class1の場所には1, それ以外は-1を入れる.
        labels = np.ones([len(categories)], dtype=np.int32) * -1
        labels[indices0] = 0
        labels[indices1] = 1
        
        self.labels = labels
        
        # 元のデータ上の位置
        indices = np.hstack([indices0, indices1])
        if data_type == DATA_TYPE_TRAIN:
            np.random.shuffle(indices)
        self.indices = indices

        if DEBUG_USE_EEG_MASK:
            # EEGのマスク例
            # FT8, FT10以外をマスクする場合
            eeg_non_mask_channel_indices = [CH_FT8, CH_FT10]
            if self.eeg_frame_type == EEG_FRAME_TYPE_FILTER:
                # Filter利用時
                self.eeg_mask = np.zeros([5, 63, 250], dtype=np.float32)
                for index in eeg_non_mask_channel_indices:
                    self.eeg_mask[:,index,:] = 1.0
            elif self.eeg_frame_type == EEG_FRAME_TYPE_FT:
                self.eeg_mask = np.zeros([17, 63, 125], dtype=np.float32)
                for index in eeg_non_mask_channel_indices:
                    self.eeg_mask[:,index,:] = 1.0
            else:
                # 通常の場合
                self.eeg_mask = np.zeros([63, 250], dtype=np.float32)
                for index in eeg_non_mask_channel_indices:
                    self.eeg_mask[index,:] = 1.0
        
        
    def get_indices(self, classify_type, categories, sub_categories, trial_mask, for_test):
        if classify_type == FACE_OBJECT:
            # Faceのindexの配列
            flags0 = [w0 and w1 for w0, w1 in \
                      zip((categories == CATEGORY_FACE),
                          (trial_mask == True))]
            # Objectのindexの配列
            flags1 = [w0 and w1 for w0, w1 in \
                      zip((categories == CATEGORY_OBJECT),
                          (trial_mask == True))]
        elif classify_type == MALE_FEMALE:
            # FaceかつMaleのindexの配列
            flags0 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_FACE),
                          (sub_categories == SUBCATEGORY_MALE),
                          (trial_mask == True))]
            # FaceかつFemaleのindexの配列
            flags1 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_FACE),
                          (sub_categories == SUBCATEGORY_FEMALE),
                          (trial_mask == True))]
        elif classify_type == ARTIFICIAL_NATURAL:
            # ObjectかつArtificialのindexの配列
            flags0 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_OBJECT),
                          (sub_categories == SUBCATEGORY_ARTIFICIAL),
                          (trial_mask == True))]
            # ObjectかつNaturalのindexの配列
            flags1 = [w0 and w1 and w2 for w0, w1, w2 in \
                      zip((categories == CATEGORY_OBJECT),
                          (sub_categories == SUBCATEGORY_NATURAL),
                          (trial_mask == True))]

        """
        if for_test and self.use_trial_average:
            # repeatの最初だけに限定する
            flags_repeat0 = self.average_repeat_indices == 0
            flags0 = flags0 & flags_repeat0
            flags1 = flags1 & flags_repeat0
        """
        
        all_indices0 = np.where(flags0)[0]
        all_indices1 = np.where(flags1)[0]
        return all_indices0, all_indices1

    def load_fmri_frame_data(self, index):
        dir_index = index // 100
        file_path = os.path.join(self.fmri_data_dir,
                                 "frames{}/frame{}.npy".format(dir_index, index))
        data = np.load(file_path)
        if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL or \
           self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
            data = np.reshape(data, [1,79,95,79])
        # (79, 95, 79)
        return data
        
    def __getitem__(self, index):
        real_index = self.indices[index]

        label = self.labels[real_index].astype(np.float32).reshape([1])

        sample = {
            'label' : label
        }

        if self.use_eeg:
            eeg_data = self.eeg_datas[real_index]
            # (63, 375) or (5, 63, 375) or (17, 63, 163)
            if self.eeg_frame_type == EEG_FRAME_TYPE_FILTER:
                # Filter利用時
                eeg_data = eeg_data[:,:,125:] # -0.5秒〜0秒までを切る場合 # (5, 63, 250)
            elif self.eeg_frame_type == EEG_FRAME_TYPE_FT:
                # FT利用時
                eeg_data = eeg_data[:,:,38:] # -0.5秒〜0秒までを切る場合 # (5, 63, 125)
            else:
                # 通常の場合
                eeg_data = eeg_data[:,125:] # -0.5秒〜0秒までを切る場合 # (63, 250)

            if DEBUG_USE_EEG_MASK:
                # マスクをかける場合
                eeg_data = eeg_data * self.eeg_mask

            sample.update( {
                'eeg_data': eeg_data,
            })
        
        if self.use_fmri:
            fmri_data = self.load_fmri_frame_data(real_index)
            sample.update( {
                'fmri_data' : fmri_data
            })
        
        return sample

    def __len__(self):
        return len(self.indices)

    @property
    def fmri_ch_size(self):
        if self.fmri_frame_type == FMRI_FRAME_TYPE_NORMAL or \
           self.fmri_frame_type == FMRI_FRAME_TYPE_AVERAGE:
            return 1
        else:
            return 3
