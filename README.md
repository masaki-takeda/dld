# Deep Learning-based Decoding of Visual Object Representation in Humans



## 1. Preface

This program was originally developed for deep neural decoding of visual object representation in human participants.

For further details, see the following papers:
- [Multimodal deep neural decoding reveals highly resolved spatiotemporal profile of visual object representation in humans](https://www.sciencedirect.com/science/article/pii/S1053811923003154)
- [Deep-learning-based fMRI decoding of real-world size for hand-held objects](https://www.biorxiv.org/content/10.64898/2026.02.01.703176v1)



## 2 Getting started

### 2.1 Docker image build

```shell
$ cd dld
$ ./scripts/build.sh
```



### 2.2. Preprocessing

When new experimental data sets are imported, the following preprocessing steps should be performed.



#### 2.2.1 Update the Excel file controlling the experimental data 

Update and save it as `dld/experiment_data/experiments.xlsx`

Note that the file should be saved in .csv format: `dld/experiment_data/experiments.csv`


In these Excel files, the experimental data must be formatted as follows. (Enter 0 for 'valid', if the data is not available.)

| valid| date | subject | run| reject_epoch|
| ------------- | ------------- | ------------- | ------------- | ------------- |
|1	|191008	|1	|1	|17,22,30,...|




#### 2.2.2 Running Docker container

Run and enter Docker container.

Navigate to the project directory.

```
$ ./scripts/run.sh
# cd /opt/dld/
```



#### 2.2.3 Data preprocessing

Convert Matlab and .csv data to numpy format.

```shell
$ ./scripts/preprocess_alpha.sh
```




| Option | Description | Choices | Default |
| ------------- | ------------- | ------------- | ------------- |
| fmri | whether to preprocess the fmri data  | "true" "false"| "true" |
| eeg | whether to preprocess the eeg data  | "true" "false"| "true" |
| eeg_frame_type | frame type of the eeg data  | "normal", "filter", "ft"| "filter" |
| eeg_duration_type | duration type of the eeg data  | "normal(1000ms)", "short(500ms)", "long(1500ms)"| "normal" |
| smooth | convert smoothed fmri data or non-smoothed data | "true" "false"| "true" |
| behavior  | whether to export a behavior data read from the csv data |"true" "false"| "true" |
| eeg_normalize_type | normalize type of the eeg data | "normal", "pre", "none" | "normal" |
| src_base | location of raw data files |  | "/data1/DLD/Data_Prepared" |
| dst_base | location of export files |  | "./data" |
| fmri_frame_type | frame type of the fmri data | "normal", "average", "three" | "normal" |
| fmri_offset_tr | TR offset of the fmri data | 1,2,3 | 2 |

Regardless of `--behavior` option, the behavior data is loaded from `experiments_data/experiemnts.csv` every time when the data is preprocessing. `--behavior` option only specifies whether to save the loaded behavior data. 

 `--eeg`, `--fmri` options do conversion as preprocessing. 



The exported files are written out to a determined subdirectory of `--dst_base`, depending on the combination specified in `--smooth` and `--fmri_frame_type` options as well as in `--eeg_normalize_type` and `--eeg_frame_type` options. For example, there is no need to separate directories for smoothing and non-smoothing; e.g., `/data2/Data_Converted_nosmoosth`. By exporting the results of conversion of both smoothing and non-smoothing data, it is possible to switch which one is loaded by the optional arguments at training. If all combinations of `--fmri_frame_type` and `--smooth` will be used for training, it must be preprocessed for such all six combinations. The same applies to the combination of `--eeg_frame_type` and `--eeg_normalize_type`.



#### 2.2.4 Trial-averaged data preprocessing

Trial-averaged data preprocessing must be done after the above data preprocessing. For the trial-averaged data preprocessing, `--average_trial_size` and `--average_repeat_size` should be added to the same options as for the normal data preprocessing.

  (The Executable file `preprocess_average.py` should have all the same options as above except for the following. It does not require the options of `--src_base` and `--behavior`.)

The same options of `--average_trial_size` and `--average_repeat_size` are used for later Training.



| Option | Description | Choices | Default |
| ------------- | ------------- | ------------- | ------------- |
| fmri | whether to preprocess the fmri data  | "true" "false"| "true" |
| eeg | whether to preprocess the eeg data  | "true" "false"| "true" |
| eeg_frame_type | frame type of the eeg data  | "normal", "filter", "ft"| "filter" |
| eeg_duration_type | duration type of the eeg data  | "normal(1000ms)", "short(500ms)", "long(1500ms)"| "normal" |
| smooth | convert smoothed fmri data or non-smoothed data | "true" "false"| "true" |
| eeg_normalize_type | normalize type of the eeg data | "normal", "pre", "none" | "normal" |
| dst_base | location of raw data and export files |  | "./data" |
| fmri_frame_type | frame type of the fmri data | "normal", "average", "three" | "normal" |
| fmri_offset_tr | TR offset of the fmri data | 1,2,3 | 2 |
| average_trial_size | average number of trials |  | 0 |
| average_repeat_size | number of repetitions for the augumentation |  | 0 |
| unmatched | whether to unmatch averaging trials | "true","false" | "false" |
| classify_type | classification type | -1,0,1,2,3,4 | -1 |


Example

```shell
python3 preprocess_average.py --dst_base=/data2/DLD/Data_Converted --eeg_frame_type=filter --average_trial_size=3 --average_repeat_size=4
```



## 3. Training

### 3.0 Learned network models
Learned network models for fMRI data, EEG data and combined fMRI-EEG data are available at:
https://drive.google.com/drive/folders/15NWHF4EsaP8PugU6GLf56-_DWv3mcIbF?usp=sharing



### 3.1 Running Docker container

```
$ ./scripts/run.sh
# cd /opt/dld/
```



### 3.2 Training EEG data

The unified entry point `main.py` is used with `--combine_type` to specify the data type.

Example

```shell
python3 main.py --combine_type=eeg --save_dir=./saved_eeg0 --data_dir=/data2/DLD/Data_Converted --model_type=tcn1 --test_subjects=TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01 --gpu=1
```

See below for options. (4.1, 4.2)



### 3.3 Training fMRI data

Example

```shell
python3 main.py --combine_type=fmri --save_dir=./saved_fmri0 --data_dir=/data2/DLD/Data_Converted --test_subjects=TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01 --gpu=1
```

See below for options. (4.1, 4.3)



### 3.4 Training Combined data

Example

```shell
python3 main.py --combine_type=combined --save_dir=./saved_combined0 --data_dir=/data2/DLD/Data_Converted --model_type=combined_tcn1 --test_subjects=TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01 --preload_eeg_dir=./saved_eeg0 --preload_fmri_dir=./saved_fmri0 --gpu=1 --lr=0.001 --lr_eeg=0.001 --lr_fmri=0.01 --weight_decay=0.0 --weight_decay_eeg=0.0 --weight_decay_fmri=0.001
```

See below for options. (4.1, 4.4)



### 3.5 Test

The `--test=true` option must be added for evaluation with test data.


Example

```shell
python3 main.py --combine_type=eeg --save_dir=./saved_eeg0 --data_dir=/data2/DLD/Data_Converted --model_type=tcn1 --test_subjects=TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01 --gpu=1 --test=true

python3 main.py --combine_type=fmri --save_dir=./saved_fmri0 --data_dir=/data2/DLD/Data_Converted --test_subjects=TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01 --test=true --gpu=1

python3 main.py --combine_type=combined --save_dir=./saved_combined0 --data_dir=/data2/DLD/Data_Converted --model_type=combined_tcn1 --test_subjects=TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01 --preload_eeg_dir=./saved_eeg0 --preload_fmri_dir=./saved_fmri0 --test=true --gpu=1
```



## 4. Training parameters

See `dld/options.py` for details.

The first argument `--combine_type` specifies which data type to use: `eeg`, `fmri`, or `combined`.



### 4.1 Options common to EEG/fMRI/Combined

| Option | Description | Choices | Default |
| ------------- | ------------- | ------------- | ------------- |
| combine_type | data type to use | "eeg", "fmri", "combined" | |
| data_seed | a random seed used for Cross-Validation separation (basically unchanged) | | 0 |
| run_seed | to fix a seed (not using a random seed), enter a specific number other than -1 (takes longer) | | -1 |
| save_dir  | save directory  | | "saved" |
| classify_type  | classification type/stimulus condition | 0=FACE/OBJECT, 1=MALE/FEMALE, 2=ARTIFICAL/NATURAL, 3=FRONT/SIDE, 4=SMALL/LARGE, -1=ALL| -1 |
| desc  | experiment descriptions | | |
| early_stopping| whether to use Early Stopping |  "true"/"false" | "true" |
| early_stopping_metric | metric used for Early Stopping evaluation | "accuracy", "roc_auc", "pr_auc", "f1", "precision", "recall", "n_precision", "n_recall", "n_f1", "n_pr_auc" | "accuracy" |
| parallel| whether to train on multiple GPUs |  "true"/"false" | "false" |
| data_dir | directory of experimental data | |  "./data" |
| eeg_normalize_type| normalize type of the eeg data　(normal=normal, pre=use the data from the period before fixations, none=no normalization) |  "normal", "pre", "none" | "normal" |
| fmri_frame_type| frame type of the fmri data (normal=normal, average=use the average data of 3TR, three=use the all data of 3TR) |  "normal", "average", "three" | "normal" |
| fmri_offset_tr | TR offset of the fmri data | 1,2,3 | 2 |
| gpu | specify the GPU to use (-1=unspecified, 0=first GPU, 1=second GPU) | | -1 |
| eeg_frame_type | frame type of the eeg data (normal=normal, filter=5ch filter, ft=FT spectrogram) | "normal", "filter", "ft" | "normal" |
| eeg_duration_type | duration type of the eeg data  | "normal(1000ms)", "short(500ms)", "long(1500ms)"| "normal" |
| smooth | whether to use smoothed fmri data | "true"/"false" | "true" |
| test_subjects | specify participants to be used for the test | enter the participants' IDs, separated by commas | "TM_200716_01,TM_200720_01,TM_200721_01,TM_200722_01,TM_200727_01" |
| test | whether for the test or not | "true"/"false" | "false" |
| fold_size | number of Fold to be used out of N Fold |  | 9 |
| subjects_per_fold | number of participants assigned to 1 Fold |  | 5 |
| patience | number of minimal continuous epochs in Early Stopping |  | 20 |	
| batch_size | batch size | | 10 |
| lr | learning rate | | 0.001 |
| weight_decay | regularization parameter weight_decay | | 0.0 |
| epochs | number of training epochs | | 100 |
| average_trial_size | average number of trials |  | 0 |
| average_repeat_size | number of repetitions for the augumentation |  | 0 |
| kernel_size | kernel size (available in TCN) |  | 3 |
| level_size | number of TemporalBlock (available in TCN) (-1=automatically calculated) |  | -1 |
| level_hidden_size | number of channels of TemporalBlock (available in TCN) (63=residual become skip connection) |  | 63 |
| residual | whether to use residual connection (available in TCN) | "true"/"false" | "true" |
| unmatched | whether to unmatch averaging trials | "true","false" | "false" |
| fmri_mask  | mask name for fMRI voxel selection  | "frontal", "occipital", "parietal", "temporal", "subcortical", "gcam", "face1" ~ "face13", "face_all", "object1" ~ "object9", "object_all" | |
| pfi_shuffle_size  | shuffle size for permutation feature importance (enabled when >=1)  | | 0 |
| debug | whether to use debug mode (use only the first few participants) | "true"/"false" | "false" |


There are two types of `fmri_mask` option usage.

- a) When  `--fmri_mask` is applied without `--pfi_shuffle_size`
  - It measures the accuracy only with some part of fMRI voxels.
  - Used for train, validation and test datasets.
- b) When  `--fmri_mask` is applied with `--pfi_shuffle_size` 
  - It measures the permutation feature importance by shuffling some part of fMRI voxels among trials.
  - Used for test of trained models.



### 4.2 Options only for EEG (--combine_type=eeg) runtime

| Option | Description | Choices | Default |
| ------------- | ------------- | ------------- | ------------- |
| model_type  | model type  | "tcn1" | "tcn1"|



### 4.3 Options only for fMRI (--combine_type=fmri) runtime

No additional options beyond the common options (4.1).



### 4.4 Options only for Combined (--combine_type=combined) runtime

| Option | Description | Choices | Default |
| ------------- | ------------- | ------------- | ------------- |
| fix_preloads | whether to retrain the weights of the preloaded EEG/fMRI model | "true" "false"| "true" |
| preload_eeg_dir |  path to the preloaded EEG model | | unspecified |
| preload_fmri_dir |  path to the preloaded fMRI model | | unspecified |
| lr_eeg | learning rate of the EEG model | | unspecified |
| lr_fmri |  learning rate of the fMRI model | | unspecified |
| weight_decay_eeg | weight decay of the EEG model | | unspecified |
| weight_decay_fmri |  weight decay of the fMRI model | | unspecified |
| model_type  | model type  | "combined_tcn1"| "combined_tcn1" |
| combined_hidden_size  | hidden size of Combined FC part | | 128 |
| combined_layer_size  | additional layer size of Combined FC part | | 0 |


`lr` specifies the learning rate for the last FC part. `lr_eeg` and `lr_fmri` specify the learning rate for the EEG and fMRI part.
`weight_decay` specifies the weight decay for the last FC part. `weight_decay_eeg` and `weight_decay_fmri` specify the weight decay for the EEG and fMRI part.

If `lr_eeg` and/or `lr_fmri` are not specified, `lr` is applied instead. `weight_decay` acts in the same manner.


If either `preload_eeg_dir` or `preload_fmri_dir` is not specified, both EEG and fMRI model are not preloaded. (Only one side is not acceptable)


Specifying `combined_hidden_size` and `combined_layer_size` can increase the additional Full Connection layer that is combined by EEG and fMRI layer.



## 6. Visualization of Grad-CAM

Output data and visualize the results as follows.

The Grad-CAM scripts automatically load training parameters from the saved `options.json` in `save_dir`. Only `--save_dir` (and optionally `--debug`) need to be specified.



Example

```shell
python3 main_grad_cam_eeg.py --save_dir=./saved_eeg0

python3 main_grad_cam_fmri.py --save_dir=./saved_fmri0
```



| Option | Description | Choices | Default |
| ------------- | ------------- | ------------- | ------------- |
| save_dir  | directory of saved model data and results | | "saved" |
| debug | whether to use debug mode | "true"/"false" | "false" |


### Output of Grad-CAM

The data will be exported to `grad_cam/data` under the `save_dir` directory in both numpy and matlab formats.



#### EEG

| Key name | Description |　Shape |
| ------------------ | ------------- | ------------- |
| guided_bp0         | Results of Guided-Backprop for label=0 | (*, 63, 250) |
| guided_bp1         | Results of Guided-Backprop for label=1 | (*, 63, 250) |
| cam_nopool0        | Results of Grad-CAM(no pooling) at each level for label=0 | (*, 7, 250) |
| cam_nopool1        | Results of Grad-CAM(no pooling) at each level for label=1 | (*, 7, 250) |
| cam0               | Results of Grad-CAM(with pooling) at each level for label=0 | (*, 7, 250) |
| cam1               | Results of Grad-CAM(with pooling) at each level for label=1 | (*, 7, 250) |
| raw_grad0          | Results of gradient(computed without effective position interpolation) at each level for label=0 | (*, 7, 63, 250) |
| raw_grad1          | Results of gradient(computed without effective position interpolation) at each level for label=1 | (*, 7, 63, 250) |
| raw_feature        | Results of activation(computed without effective position interpolation) at each level | (*, 7, 63, 250) |
| flat_active_grad0  | Results of gradient(computed only from effective position in dilated conv) at each level for level=0 | (*, 7, []) |
| flat_active_grad1  | Results of gradient(computed only from effective position in dilated conv) at each level for level=1 | (*, 7, []) |
| flat_active_feature| Results of activation(computed only from effective position in dilated conv) at each level | (*, 7, []) |
| label              | correct label (0 or 1) | (1, *) |
| predicted_label    | predicted label ((predicted_prob>0.5): 1, otherwise: 0) | (1, *) |
| predicted_prob     | predicted probability (0.0~1.0) | (1, *) |

(*`7` in the above list are examples for `kernel_size=2`. If the `kernel_size` changes, so will the values.)

To calculate Guided-Grad-CAM, input `cam_nopool0 * guided_bp0` or `cam_nopool1 * guided_bp1`.

`cam_nopool0`, `cam_nopool1`, `cam0`, and `cam1` are interpolated values to make gradient x activation 250 frames. The gradient x activation are computed based on `flat_active_grad0`, `flat_active_grad1`, and `flat_active_feature`. 

Although `flat_active_grad0`, `flat_active_grad1`, and `flat_active_feature` have different number of elements at each level, they are flattened into a one-dimensional array to fit the different number of elements into an array.
For example, the number of levels is 7 when kernel_size=2. The active gradient (i.e., effective gradient) have a one-dimensional array: `(15750), (7875), (3969), (1953), (945), (441), (189)` in which `(63, 250), (63, 125), (63, 63), (63, 31), (63, 15), (63, 7), (63, 3)` are flattened.



#### fMRI


| Key name | Description |　Shape |
| ------------------ | ------------- | ------------- |
| cam0               | Results of final layer Grad-CAM with Global-Pooling for label=0| (*, 6, 7, 6)  |
| cam1               | Results of final layer Grad-CAM with Global-Pooling for label=1 | (*, 6, 7, 6)  |
| cam_nopool0        | Results of final layer Grad-CAM without Global-Pooling for label=0 | (*, 6, 7, 6) |
| cam_nopool1        | Results of final layer Grad-CAM without Global-Pooling for label=1 | (*, 6, 7, 6) |
| guided_bp0         | Results of Guided-Backprop for label=0 | (*, 79, 95, 79) |
| guided_bp1         | Results of Guided-Backprop for label=1 | (*, 79, 95, 79) |
| guided_cam_nopool0 | Results of Guided-Grad-CAM without Global-Pooling for label=0 | (*, 79, 95, 79) |
| guided_cam_nopool1 | Results of Guided-Grad-CAM without Global-Pooling for label=1 | (*, 79, 95, 79) |
| label              | correct label (0 or 1) | (1, *) |
| predicted_label    | predicted label ((predicted_prob>0.5): 1, otherwise: 0) | (1, *) |
| predicted_prob     | predicted probability (0.0~1.0) | (1, *) |

When `label` and `predicted_label` show 1, the prediction is correct and `guided_cam_nopool1` indicates the result of Guided-Grad-CAM when the label=1 (i.e., correct answer) is predicted.

When `label` shows 1 and `predicted_label` shows 0, the prediction is wrong.



#### Combined

| Key name              | Description                                                  | Shape           |
| --------------------- | ------------------------------------------------------------ | --------------- |
| e_guided_bp0          | EEG results of Guided-Backprop for label=0 | (*, 79, 95, 79)               |
| e_guided_bp1          | EEG results of Guided-Backprop for label=1 | (*, 79, 95, 79)               |
| e_cam_nopool0         | EEG results of Grad-CAM(no pooling) at each level for label=0    | (*, 7, 250)     |
| e_cam_nopool1         | EEG results of Grad-CAM(no pooling) at each level for label=1    | (*, 7, 250)     |
| e_cam0                | EEG results of final layer Grad-CAM with Global-Pooling for label=0    | (*, 7, 250)     |
| e_cam1                | EEG results of final layer Grad-CAM with Global-Pooling for label=1   | (*, 7, 250)     |
| e_raw_grad0           | EEG results of gradient(computed without effective position interpolation) at each level for label=0 | (*, 7, 63, 250) |
| e_raw_grad1           | EEG results of gradient(computed without effective position interpolation) at each level for label=1 | (*, 7, 63, 250) |
| e_raw_feature         | EEG results of activation(computed without effective position interpolation) at each level          | (*, 7, 63, 250) |
| e_flat_active_grad0   | EEG results of gradient(computed only from effective position in dilated conv) at each level for level=0 | (*, 7, [])      |
| e_flat_active_grad1   | EEG results of gradient(computed only from effective position in dilated conv) at each level for level=1 | (*, 7, [])      |
| e_flat_active_feature | EEG results of activation(computed only from effective position in dilated conv) at each level | (*, 7, [])      |
| f_cam0                | fMRI results of final layer Grad-CAM with Global-Pooling for label=0 | (*, 6, 7, 6)    |
| f_cam1                | fMRI results of final layer Grad-CAM with Global-Pooling for label=1 | (*, 6, 7, 6)    |
| f_cam_nopool0         | fMRI results of final layer Grad-CAM without Global-Pooling for label=0 | (*, 6, 7, 6)    |
| f_cam_nopool1         | fMRI results of final layer Grad-CAM without Global-Pooling for label=1 | (*, 6, 7, 6)    |
| f_guided_bp0          | fMRI results of Guided-Backprop for label=0              | (*, 79, 95, 79) |
| f_guided_bp1          | fMRI results of Guided-Backprop for label=1              | (*, 79, 95, 79) |
| f_guided_cam_nopool0  | fMRI results of Guided-Grad-CAM without Global-Pooling for label=0 | (*, 79, 95, 79) |
| f_guided_cam_nopool1  | fMRI Results of Guided-Grad-CAM without Global-Pooling for label=1 | (*, 79, 95, 79) |
| label              | correct label (0 or 1) | (1, *) |
| predicted_label    | predicted label ((predicted_prob>0.5): 1, otherwise: 0) | (1, *) |
| predicted_prob     | predicted probability (0.0~1.0) | (1, *) |



## 	8. Hyperparameter optimization

Hyperparameter optimization is performed using Bayesian optimization with Optuna. Prepare a script referring to `main_optimize_sample.py` and run it.


In the script, specify the executable file, save directory name prefix, optimization target options, and fixed options as follows.

```python
#-----[Changes from here]-----
# Trail size for hyper-parameter optization
optimization_trial_size = 10

# Combine type
combine_type = 'eeg' # combined, fmri, eeg
classify_type = FRONT_SIDE # FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL, FRONT_SIDE, SMALL_LARGE

save_dir_prefix = "./saved_eeg_op0/op"
# When specifying up to a subdirectory as above
# The save directory will be "./saved_eeg_op0/op_0", "./saved_eeg_op0/op_1" ... and so on

def objective_func(trial):
    params = {
        # Optimizing parameters
        'lr'          : trial.suggest_loguniform('lr', 0.0001, 0.1),
        #'weight_decay': trial.suggest_uniform('weight_decay', 0.0, 0.1),
        'kernel_size' : trial.suggest_int('kernel_size', 2, 9),
        #'residual'    : trial.suggest_categorical('residual', [0, 1]),
        # Non-numeric values such as model type can also be specified with categorical
        #'model_type'  : trial.suggest_categorical('model_type', ['tcn1', 'tcnx']),

        # Fixed parameters
        'eeg_normalize_type' : 'pre',
        'batch_size' : 100,
        'patience'   : 50,
        'data_dir'   : '/data2/DLD/Data_Converted',
        'fold_size'  : 1, # When only 1 Fold is targeted to speed up the processing
        'gpu'        : 1,
    }
    return params
#-----[To here]-----
```


Prepare and run a script in the same manner as the sample.

```shell
python3 main_optimize_sample.py
```



In the above example, after execution, a summary is exported to `./saved_eeg_op0/op_summary/summary.txt` as follows.

(The one with the best validation score will be labeled **best**.)

Also, the parameters for the best validation score are written at the top.



```
[Best params]
  lr = 0.00015342044981142149
  kernel_size = 7

[Results]
  [    ] ./saved_eeg_op_av1/op_0: 60.90794451450189 lr=0.0002, kernel_size=4,
  [    ] ./saved_eeg_op_av1/op_1: 61.7906683480454 lr=0.0004, kernel_size=9,
  [    ] ./saved_eeg_op_av1/op_2: 59.01639344262295 lr=0.0003, kernel_size=3,
  [    ] ./saved_eeg_op_av1/op_3: 62.42118537200504 lr=0.0028, kernel_size=4,
  [best] ./saved_eeg_op_av1/op_4: 63.68221941992434 lr=0.0002, kernel_size=7,
  [    ] ./saved_eeg_op_av1/op_5: 61.53846153846154 lr=0.0017, kernel_size=7,
  [    ] ./saved_eeg_op_av1/op_6: 62.042875157629254 lr=0.0014, kernel_size=9,
  [    ] ./saved_eeg_op_av1/op_7: 50.94577553593947 lr=0.0851, kernel_size=6,
  [    ] ./saved_eeg_op_av1/op_8: 59.52080706179067 lr=0.0041, kernel_size=3,
  [    ] ./saved_eeg_op_av1/op_9: 60.529634300126105 lr=0.0003, kernel_size=6,
  [    ] ./saved_eeg_op_av1/op_10: 49.05422446406053 lr=0.0215, kernel_size=8,
  [    ] ./saved_eeg_op_av1/op_11: 59.89911727616646 lr=0.0073, kernel_size=5,
...
```



To perform test evaluation on the optimized model, add `--test=true` and run.

```
python3 main_optimize_sample.py --test=true
```

The evaluation results will be recorded in `./saved_eeg_op0/op_summary/summary_test.txt`.

