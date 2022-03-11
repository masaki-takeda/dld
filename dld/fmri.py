import numpy as np
import os
from scipy import io
import nibabel as nib


from behavior import Behavior


class FMRI(object):
    def __init__(self,
                 src_base,
                 behavior,
                 normalize_per_run=True,
                 frame_type="normal",
                 smooth=True):
        # 連続3フレームの平均を取るかどうか
        assert (frame_type == "normal" or frame_type == "average" or frame_type == "three")
        
        self.frame_type = frame_type
        self.smooth = smooth

        if smooth:
            # smoothingをかけた場合
            prefix = "mswuep"
        else:
            # smoothingをかけていない場合
            prefix = "mwuep"

        path_format = "{0}/MRI/TM_{1}_{2:0>2}/work_4D/{3}{4:0>2}_4D.nii"
        
        path = path_format.format(src_base,
                                  behavior.date,
                                  behavior.subject,
                                  prefix,
                                  behavior.run)
        
        nii = nib.load(path)
        fmri_datas = np.array(nii.dataobj)
        # (79, 95, 79, 262)
        fmri_datas = np.transpose(fmri_datas, [3, 2, 1, 0])
        # (262, 79, 95, 79) float64
        # (t,z,y,x)

        # ノーマライズ
        if normalize_per_run:
            fmri_datas = self.normalize_data(fmri_datas)

        # +4秒後(+2TR)の部分をとってくる.
        # (正確には+4sec〜+6secの間のフレームを採用する)
        if frame_type == "normal":
            frame_indices = behavior.get_fmri_tr_indices(offset_tr=2)
            self.data = fmri_datas[frame_indices,:,:,:].astype(np.float32)
            # (37, 79, 95, 79)等
        elif frame_type == "average":
            frame_indices = behavior.get_fmri_successive_tr_indices(
                offset_tr=2,
                successive_frames=3)
            frame_indices = np.array(frame_indices, dtype=np.int32)
            org_data = fmri_datas[frame_indices].astype(np.float32)
            # (37, 3, 79, 95, 79)等
            self.data = np.mean(org_data, axis=1)
        else:
            # frame_type="three"の倍
            frame_indices = behavior.get_fmri_successive_tr_indices(
                offset_tr=2,
                successive_frames=3)
            frame_indices = np.array(frame_indices, dtype=np.int32)
            self.data = fmri_datas[frame_indices].astype(np.float32)
            # (37, 3, 79, 95, 79)等
        
        del nii
        del fmri_datas
        
    def normalize_data(self, fmri_datas):
        fmri_mean = fmri_datas.mean(axis=0)
        fmri_std  = fmri_datas.std(axis=0)
        
        EPSILON = 0.00001
        fmri_std = np.maximum(fmri_std, EPSILON)
        fmri_norm_datas = (fmri_datas - fmri_mean) / fmri_std
        return fmri_norm_datas

    @property
    def frame_size(self):
        return len(self.data)

    def export(self, dst_base, start_frame_index):
        if self.frame_type == "normal":
            # 通常の場合
            parent_dir_path = os.path.join(dst_base, "final_fmri_data")
        elif self.frame_type == "average":
            # 連続3フレームの平均を取る場合
            parent_dir_path = os.path.join(dst_base, "final_fmri_data_av")
        else:
            # 連続3フレームを使う場合場合
            parent_dir_path = os.path.join(dst_base, "final_fmri_data_th")

        if not self.smooth:
            # smoothingをかけていない場合に_nsをお尻に付加する.
            parent_dir_path = parent_dir_path + "_nosmooth"
            
        if not os.path.exists(parent_dir_path):
            os.mkdir(parent_dir_path)

        for i in range(len(self.data)):
            frame_index = start_frame_index + i
            
            # 100個ずつ保存ディレクトリを分ける
            dir_name = "frames{}".format(frame_index // 100)
            dir_path = os.path.join(parent_dir_path, dir_name)

            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            file_path = os.path.join(dir_path, "frame{}".format(frame_index))
            frame_data = self.data[i]
            np.save(file_path, frame_data)
