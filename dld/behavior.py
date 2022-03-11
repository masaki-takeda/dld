import numpy as np
try:
    import h5py
except:
    print('h5py import error')


class Trial(object):
    def __init__(self,
                 category,
                 identity,
                 angle,
                 time,
                 trial_id):
        self.category = category # 1,2
        self.identity = identity # 1,2,3,4
        self.angle    = angle    # 1~5
        self.time     = time     # float
        self.trial_id = trial_id # 1~
        
    @property
    def tr(self):
        return int(round(self.time)) // 2

    @property
    def trial_index(self):
        """ 0始まりのindex """
        return self.trial_id - 1

    @property
    def sub_category(self):
        if self.identity == 1 or self.identity == 2:
            # Male or Artificial
            return 1
        else:
            # Female or Natural
            return 2


class Behavior(object):
    """ PshchToolboxデータ. (preprocess時に利用) """
    
    def __init__(self, src_base, date, subject, run, reject_trials):
        #suffix = ""
        # 反応時間が2秒を超えたトライアルを除外したバージョン
        suffix = "_2sdelete"
        
        mat_path = "{0}/PsychToolbox/TM_{1}_{2:0>2}/TM_{1}_{2:0>2}_{3:0>2}{4}.mat".format(
            src_base, date, subject, run, suffix)
        
        csv_path = "{0}/PsychToolbox/TM_{1}_{2:0>2}/TM_{1}_{2:0>2}_{3:0>2}{4}.csv".format(
            src_base, date, subject, run, suffix)
        
        self.date    = date    # 191008など
        self.subject = subject # 1, 2
        self.run     = run

        # MATLAB ver7の形式の場合
        #mat_all_data = io.loadmat(mat_path)
        #head_time = mat_all_data['time']['TR'][0][0][0][0]

        # MATLAB ver7.3の形式の場合
        with h5py.File(mat_path,'r') as f:
            head_time = f['time']['TR'][0][0]
        
        csv_data = np.loadtxt(csv_path,
                              delimiter=",",
                              skiprows=1,
                              usecols=(3,4,5,13,16,17))
        # (50, 6)
        # Categ, Identity, Angle, IMGtime, Correct, Trial
        
        # IMGtimeから先頭時刻と10秒(=5TR)を引いておく
        csv_data[:,3] -= (head_time + 10.0)

        # Identityはサブカテゴリを表し
        # Categ=1(Face)の場合は1,2が男、3,4が女 (M1, M2, F1, F2)
        # Categ=2(Object)の場合は1,2がArtificial, 3,4がNatural)
        
        trials = []
        
        for csv_row in csv_data:
            category = int(csv_row[0])
            identity = int(csv_row[1])
            angle    = int(csv_row[2])
            time     = csv_row[3]
            correct  = int(csv_row[4]) == 1
            trial_id = int(csv_row[5])
            
            if (category != 0) and correct and (trial_id not in reject_trials):
                # Fixation trialを除外
                # 正解しなかったtrialを除外
                # Reject trialを除外
                trial = Trial(category, identity, angle, time, trial_id)
                trials.append(trial)
        
        self.trials = trials
        
    @property
    def subject_id(self):
        # "TM_191008_01" の形式の被験者IDを返す.
        return "TM_{0}_{1:0>2}".format(self.date, self.subject)
        
    @property
    def trial_indices(self):
        return [ trial.trial_index for trial in self.trials ]

    def get_fmri_tr_indices(self, offset_tr):
        return [ trial.tr + offset_tr for trial in self.trials ]

    def get_fmri_successive_tr_indices(self, offset_tr, successive_frames):
        return [ [i for i in range(trial.tr + offset_tr, trial.tr + offset_tr+successive_frames)]
                  for trial in self.trials ]



if __name__ == '__main__':
    behavior = Behavior("./data1/DLD/Data_Prepared",
                        191008,
                        1,
                        1,
                        [])
    
