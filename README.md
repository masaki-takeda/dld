# Multi-modal deep neural decoding of visual object representation in humans

Getting started


## 1. Docker image build

```shell
$ cd dld
$ ./scripts/build.sh
```



## 2. Preprocessing

It must be performed the following steps for preprocessing when new experimental data are added or the preprocessing method is changed.



### 2.1 Update the Excel file controling experimental data 

Update and save as `dld/experiment_data/experiments.xlsx`

Save also in .csv format as `dld/experiment_data/experiments.csv`


In these Excel files, the experimental data must be formatted as follows. (Enter 0 for 'valid', if the data is not available.)

| valid| date | subject | run| reject_epoch|
| ------------- | ------------- | ------------- | ------------- | ------------- |
|1	|191008	|1	|1	|17,22,30,...|




### 2.2 Running Docker container

Run and enter Docker container.

Navigate to the project directory.

```
$ ./scripts/run.sh
# cd /opt/simul/
```



### 2.3 Data preprocessing

Convert Matlab and csv data to numpy format.

```shell
$ ./scripts/preprocess_alpha.sh
```




| Option | Description | Choices | Default |
| ------------- | ------------- | ------------- | ------------- |
| fmri | whether to preprocess the fmri data  | "true" "false"| "true" |
| eeg | whether to preprocess the eeg data  | "true" "false"| "true" |
| eeg_frame_type | frame type of the eeg data  | "normal", "filter", "ft"| "filter" |
| smooth | whether to smooth the fmri data | "true" "false"| "true" |
| behavior  | whether to export a behavior data read from the csv data |"true" "false"| "true" |
| eeg_normalize_type | normalize type of the eeg data | "normal", "pre", "none" | "normal" |
| src_base | location of raw data files |  | "/data1/DLD/Data_Prepared" |
| dst_base | location of export files |  | "./data" |
| fmri_frame_type | frame type of the fmri data | "normal", "average", "three" | "normal" |

`--behavior` Regardless of this option, the behavior data is loaded from `experiments_data/experiemnts.csv` every time when the data preprocessing. `--behavior` option only specifies whether to save the loaded behavior data. 

 `--eeg`, `--fmri` options do conversion as preprocessing. #or# These options determine whether conversion is done.



The export files are written out to a determined subdirectory of `--dst_base`, depending on the combination specified in `--smooth` and `--fmri_frame_type` options, and `--eeg_normalize_type` and `--eeg_frame_type` options. For example, there is no need to separate directories for smoothing and non-smoothing; e.g., `/data2/Data_Converted_nosmoosth`. By placing the results of conversion of both smoothing and non-smoothing data, it is possible to switch which one is loaded by the optional arguments at Training. If all combinations of '--fmri_frame_type' and '--smooth' will be used for Training, it must be preprocessed for such all six combinations. The same applies to the combination of '--eeg_frame_type' and '--eeg_normalize_type'.



### 2.4 Trial-averaged data preprocessing

Trial-averaged data preprocessing must be done after the normal data preprocessing. For the trial-averaged data preprocessing, `--average_trial_size` and `--average_repeat_size` should be added to the same options as for the normal data preprocessing.

  (The Executable file should have all the same options as ### except for the following. The file does not require the options of`--src_base` and `--behvavior` unlike `preprocess_average.py`)


後の学習時にも同じ `--average_trial_size`, `--average_repeat_size` オプションを利用する.


| Option | 内容 | 選択肢 |デフォルト選択肢|
| ------------- | ------------- | ------------- | ------------- |
| fmri | fMRIのデータをpreprocessするかどうか  | "true" "false"| "true" |
| eeg | EEGのデータをpreprocessするかどうか  | "true" "false"| "true" |
| eeg_frame_type | EEGのフレームタイプ  | "normal", "filter", "ft"| "filter" |
| smooth | smoothingされたfMRIのデータをコンバート対象とするかどうか | "true" "false"| "true" |
| eeg_normalize_type | EEGのノーマライズタイプ | "normal", "pre", "none" | "normal" |
| dst_base | 読み込みと書き出しの場所 |  | "./data" |
| fmri_frame_type | fMRIのフレームタイプ | "normal", "average", "three" | "normal" |
| average_trial_size | 平均Trial数 |  | 0 |
| average_repeat_size | データ水増しの繰り返し数 |  | 0 |



実行例

```shell
python3 preprocess_average.py --dst_base=/data2/DLD/Data_Converted --eeg_frame_type=filter --average_trial_size=3 --average_repeat_size=4
```



## 3. 学習

### 3.1 Docker container起動

```
$ ./scripts/run.sh
# cd /opt/simul/
```



### 3.2 EEGデータの学習

実行例

```shell
python3 main_eeg.py --save_dir=./saved_eeg0 --data_dir=/data2/DLD/Data_Converted --model_type=tcn1 --test_subjects=TM_191008_01,TM_191009_01 --gpu=1
```

オプションに関しては下で後述



### 3.3 fMRIデータの学習

実行例

```shell
python3 main_fmri.py --save_dir=./saved_fmri0 --data_dir=/data2/DLD/Data_Converted --test_subjects=TM_191008_01,TM_191009_01 --gpu=1
```



### 3.4 Combinedデータの学習

実行例

```shell
python3 main_combined.py --save_dir=./saved_combined0 --data_dir=/data2/DLD/Data_Converted --model_type=combined_tcn1 --test_subjects=TM_191008_01,TM_191009_01 --preload_eeg_dir=./saved_eeg0 --preload_fmri_dir=./saved_fmri0 --gpu=1 --lr=0.001 --lr_eeg=0.001 --lr_fmri=0.01 --weight_decay=0.0 --weight_decay_eeg=0.0 --weight_decay_fmri=0.001
```



### 3.5 テスト評価

テストデータを用いた評価を行うには`--test=true`オプションを付加する.

実行例

```shell
python3 main_eeg.py --save_dir=./saved_eeg0 --data_dir=/data2/DLD/Data_Converted --model_type=tcn1 --test_subjects=TM_191008_01,TM_191009_01 --gpu=1 --test=true

python3 main_fmri.py --save_dir=./saved_fmri0 --data_dir=/data2/DLD/Data_Converted --test_subjects=TM_191008_01,TM_191009_01 --test=true --gpu=1

python3 main_combined.py --save_dir=./saved_combined0 --data_dir=/data2/DLD/Data_Converted --model_type=combined_tcn1 --test_subjects=TM_191008_01,TM_191009_01 --preload_eeg_dir=./saved_eeg0 --preload_fmri_dir=./saved_fmri0 --test=true --gpu=1
```



### 3.6 Out out memory対策

1枚目のGPUのGPUメモリが他プロセスによって占有されている場合、GPUオプション`--gpu=1`と明示的に指定している場合でも、`RuntimeError: CUDA error: out of memory`　というエラーが出てしまう場合がある.

その場合は、以下の様に`export export CUDA_VISIBLE_DEVICES=1`の環境変数を設定した上で、`--gpu=0`を指定して実行することで回避を行う.

(プログラム側からは2枚目のGPUのみが見えていて、これが1枚目のGPUとして見えるように指定している)

```shell
$ export CUDA_VISIBLE_DEVICES=1
$ python3 main_combined.py --save_dir=./saved_combined0 --data_dir=/data2/DLD/Data_Converted --model_type=combined_filter1 --test_subjects=TM_191008_01,TM_191009_01 --preload_eeg_dir=./saved_eeg0 --preload_fmri_dir=./saved_fmri0 --gpu=0
```



### 3.7 モデルタイプ

現在実装済みの `model_type`引数は以下の通りである。



#### EEG

| model_type | モデル名 | 内容 |
| :--------- | ------------------ | ---- |
| model1     | EEGModel           |1次元Convを使ったモデル  |
| model2     | EEGModel2          |EEGModelの1層目のstrideを大きくしたもの |
| rnn1       | EEGRNNModel        |RNN(LSTM)を利用したもの   |
| convrnn1   | EEGConvRNNModel    |1次元Convの後にLSTMを付加したもの |
| filter1    | EEGFilterModel     |横:250, 縦:63, 色:5chの画像と同様なものとして2次元の畳み込みを行ったFilter対応モデル      |
| filter2    | EEGFilterModel2    | 横:250, 縦:5, 色:63chの画像と同様なものとして2次元の畳み込みを行ったFilter対応モデル     |
| filter3    | EEGFilterModel3    | 一番最初にFullコネクト層を入れた後に1次元Convを入れたFilter対応モデル  |
| ft1        | EEGFtModel         | FT Spectrogram対応モデル |
| stnn1      | EEGSTNNModel       | STNN model |
| tcn1      | EEGSTCNModel       | TCN model 最終stepのみ利用 |
| tcn2      | EEGSTCNModel2       | TCN model 全step利用 |

#### fMRI

| model_type | モデル名 | 内容 |
| :--------- | ------------------ | ---- |
|            | FMRIModel          |3次元Convを使ったモデル  |

#### Combined

| model_type | モデル名 | 内容 |
| :--------- | -------- | ---- |
| combined1     | CombinedModel |  FMRIModelとEEGModel(`model1`)に対応したCombinedモデル    |
| combined_filter1     |  CombinedFilterModel   | FMRIModelとEEGFilterModel2(`filter2`)に対応したFilter対応モデル  |
| combined_tcn1     |  CombinedTCNModel   | FMRIModelとEEGTCNModel(`tcn1`)に対応したモデル  |




## 4. 学習用パラメータ

詳細は`dld/options.py` にて



### 4.1 EEG/fMRI/Combined共通オプション

| Option | 内容 | 選択肢 |デフォルト選択肢|
| ------------- | ------------- | ------------- | ------------- |
| data_seed | データセットのCrossValidation分割に使用する乱数のseed(基本変更しない) | | 0 |
| run_seed | 実行時の乱数を固定する場合に-1以外を指定する(指定すると実行は遅くなる) | | -1 |
| save_dir  | 保存ディレクトリ  | | "saved" |
| classify_type  | 分類タイプ  | 0=FACE/PLACE 1=MALE/FEMALE, 2=ARTIFICAL/NATURAL, -1=ALL| -1 |
| desc  | 実験内容  | | |
| early_stopping| Early Stopping利用するかどうか|  "true"/"false" | "true" |
| parallel| マルチGPUで学習するかどうか |  "true"/"false" | "false" |
| data_dir | 実験データのディレクトリ | |  "./data" |
| eeg_normalize_type| EEGのノーマライズタイプ(nomal=通常, pre=fixation前の期間を利用, none=ノーマライズ無し|  "normal", "pre", "none" | "normal" |
| fmri_frame_type| fMRIのフレームタイプ(nomal=通常, average=3TRの平均, three=3TRを全部利用)|  "normal", "avarage", "three" | "normal" |
| gpu | 利用するGPU指定(-1なら無指定,0ならGPU1枚目, 1ならGPU2枚目) | | -1 |
| eeg_frame_type | EEGのフレームタイプ(normal=通常, filter=5chフィルタ, ft=FT spectorogram) | "normal", "filter", "ft" | "filter" |
| smooth | fMRIにてsmoothingデータを利用するかどうか | "true"/"false" | "true" |
| test_subjects | test用に指定する被験者 | 被験者のIDをカンマ区切りで指定 | "TM_191008_01,TM_191009_01" |
| test | テスト評価かどうか | "true"/"false" | "false" |
| fold_size | 10Foldの内の利用するFold数 |  | 10 |
| subjects_per_fold | 1Foldに割り当てる被験者数 |  | 4 |
| patience | Early stoppingの継続epoch数 |  | 20 |
| batch_size | バッチサイズ | | 10 |
| lr | 学習率 | | 0.001 |
| weight_decay | weight_decay正則化パラメータ | | 0.0 |
| epochs | 学習Epoch数 | | 100 |
| average_trial_size | 平均Trial数 |  | 0 |
| average_repeat_size | データ水増しの繰り返し数 |  | 0 |
| kernel_size | kernel size (STNN,TCNでのみ有効) |  | 3 |
| level_size | TemporalBlock数 (TCNでのみ有効). -1なら自動で算出 |  | -1 |
| level_hidden_size | TemporalBlockのch数 (TCNでのみ有効). 63ならresidualがskip接続になる |  | 63 |
| residual | residual connectionを使うかどうか(TCNでのみ有効) | "true"/"false" | "true" |



### 4.2 main_eeg.py実行時のみのオプション

| Option | 内容 | 選択肢 |デフォルト選択肢|
| ------------- | ------------- | ------------- | ------------- |
| model_type  | モデルタイプ  | "model1", "model1", "rnn1", "convrnn1", "filter1", "filter2", "filter3", "stnn1", "tcn1", "tcn2"| "model1"|



### 4.3 main_combined.py実行時のみのオプション

| Option | 内容 | 選択肢 |デフォルト選択肢|
| ------------- | ------------- | ------------- | ------------- |
| fix_preloads |  PreloadしたEEG/fMRIモデルの重みを固定するか再学習するかどうか | "true" "false"| "true" |
| preload_eeg_dir |  PreloadするEEGモデルのパス | | 無指定 |
| preload_fmri_dir |  PreloadするfMRIモデルのパス | | 無指定 |
| lr_eeg |  EEGモデル部分のlr | | 無指定 |
| lr_fmri |  fMRI部分の学習率 | | 無指定 |
| weight_decay_eeg |  EEGモデル部分のweight decay | | 無指定 |
| weight_decay_fmri |  fMRI部分のweight decay | | 無指定 |
| model_type  | モデルタイプ  | "combined1", "combined_filter1", "combined_tcn1"| "combined_tcn1" |
| combined_hidden_size  | Combined FC部分のhiddenサイズ| | 128 |
| combined_layer_size  | Combined FC部分の追加層数 | | 0 |


`lr`は、最後のFC部分の学習率の指定. `lr_eeg`, `lr_fmri`はEEG, fMRI部分の学習率の指定.
`weight_decay`は、最後のFC部分のweight decayの指定. `weight_decay_eeg`, `weight_decay_fmri`はEEG, fMRI部分のweight decayの指定.

`lr_eeg`, `lr_fmri` を指定しなかった場合は、`lr` が代わりにその部分に指定される. `weight_decay` も同様.


`preload_eeg_dir`もしくは、`preload_frmi_dir`のどちらかを指定しなかった場合はEEG, fMRIともにpreloadはされない. (片方のみのprelaod指定は不可)


`combined_hidden_size` および `combined_layer_size` を指定することでEEG, fMRI層の結合後のFull Connect層の追加分を増やすことができる. `combined_hidden_size=128`, `combined_layer_size=0`にすると従来と同じ内容となる.



## 6. Grad-CAM可視化

Grad-CAMの可視化は、データの出力と、結果データの可視化を以下の手順で行う.



実行例:


```shell
python3 main_grad_cam_eeg.py --save_dir=./saved_eeg0 --data_dir=/data2/DLD/Data_Converted --model_type=tcn1 --test_subjects=TM_191008_01,TM_191009_01 --gpu=1 --test=true

python3 main_grad_cam_fmri.py --save_dir=./saved_fmri0 --data_dir=/data2/DLD/Data_Converted --test_subjects=TM_191008_01,TM_191009_01 --gpu=1 --test=true
```

Grad-CAM計算時は基本的に**test時**に使ったオプションと同じものを使い、そこから`run_seed`, `classify_type`, `early_stopping`, `parallel`, `patience`, `batch_size` を抜いたものを使う). また、`main_grad_cam_combined` 実行時には、学習時と異なり `preload_eeg_dir`, `preload_fmri_dir` は不要である.



| Option | 内容 | 選択肢 |デフォルト選択肢|
| ------------- | ------------- | ------------- | ------------- |
| data_dir  | データディレクトリ  | | "./data" |
| save_dir  | モデルデータおよび結果出力ディレクトリ  | | "saved" |
| classify_type  | 分類タイプ  | 0=FACE/PLACE 1=MALE/FEMALE, 2=ARTIFICAL/NATURAL, -1=ALL| -1 |
| eeg_normalize_type| EEGのノーマライズタイプ(nomal=通常, pre=fixation前の期間を利用, none=ノーマライズ無し|  "normal", "pre", "none" | "normal" |
| fmri_frame_type| fMRIのフレームタイプ(nomal=通常, average=3TRの平均, three=3TRを全部利用)|  "normal", "avarage", "three" | "normal" |
| model_type  | モデルタイプ  | 学習時と同じmodel_typeを指定する | 学習時と同じmodel_typeを指定する |
| eeg_frame_type | EEGのフレームタイプ(normal=通常, filter=5chフィルタ, ft=FT spectorogram) | "normal", "filter", "ft" | "filter" |
| smooth | fMRIにてsmoothingデータを利用するかどうか | "true"/"false" | "true" |
| gpu | 利用するGPU指定(-1なら無指定,0ならGPU1枚目, 1ならGPU2枚目) | | -1 |
| data_seed | データセットのCrossValidation分割に使用する乱数のseed(基本変更しない) | | 0 |
| test | テストデータセットを利用するかどうか(falseならvlidationデータセット) | "true"/"false" | "true" |
| test_subjects | test用に指定する被験者 | 被験者のIDをカンマ区切りで指定 | "TM_191008_01,TM_191009_01" |
| fold_size | 10Foldの内の利用するFold数 |  | 10 |
| subjects_per_fold | 1Foldに割り当てる被験者数 |  | 4 |
| kernel_size | kernel size (STNN,TCNでのみ有効) |  | 3 |
| level_size | num_channelsの数 (TCNでのみ有効). -1なら自動で算出 |  | -1 |
| level_hidden_size | num_channelsのch数 (TCNでのみ有効). |  | 63 |
| residual | residual connectionを使うかどうか(TCNでのみ有効) | "true"/"false" | "true" |



### Grad-CAMデータ形式

`save_dir`で指定したディレクトリ以下の、`grad_cam/data`以下に、numpy形式とmatlab形式の両方で書き出される.



#### EEG

| Key名 | 内容 |　Shape |
| ------------------ | ------------- | ------------- |
| guided_bp0         | Guided-Backpropの結果, label=0に対する結果 | (*, 63, 250) |
| guided_bp1         | Guided-Backpropの結果, label=1に対する結果 | (*, 63, 250) |
| cam_nopool0        | 各LevelでのGrad-CAM(poolingなし), label=0に対する結果 | (*, 7, 250) |
| cam_nopool1        | 各LevelでのGrad-CAM(poolingなし), label=1に対する結果 | (*, 7, 250) |
| cam0               | 各LevelでのGrad-CAM(pooling有り), label=0に対する結果 | (*, 7, 250) |
| cam1               | 各LevelでのGrad-CAM(pooling有り), label=1に対する結果 | (*, 7, 250) |
| raw_grad0          | 各Levelでの(有効位置を考慮しない生の)勾配, label=0に対する結果 | (*, 7, 63, 250) |
| raw_grad1          | 各Levelでの(有効位置を考慮しない生の)勾配, label=1に対する結果 | (*, 7, 63, 250) |
| raw_feature        | 各Levelでの(有効位置を考慮しない生の)Activation        | (*, 7, 63, 250) |
| flat_active_grad0  | 各Levelでのdilated convでの有効位置のみを抜き出した勾配, level=0に対する結果 | (*, 7, []) |
| flat_active_grad1  | 各Levelでのdilated convでの有効位置のみを抜き出した勾配, level=1に対する結果 | (*, 7, []) |
| flat_active_feature| 各Levelでのdilated convでの有効位置のみを抜き出したActivation | (*, 7, []) |
| label              | 正解ラベル (0 or 1) | (1, *) |
| predicted_label    | 予測ラベル (predicted_prob > 0.5なら1, そうでなければ0) | (1, *) |
| predicted_prob     | 予測確率 (0.0~1.0) | (1, *) |

(※リスト内の`7` の部分は`kernel_size=2` の場合の例で、`kernel_size` を変えると値は変わる)

Guided-Grad-CAMを算出するには、`cam_nopool0 * guided_bp0` もしくは `cam_nopool1 * guided_bp1` を利用する.
`cam_nopool0`, `cam_nopool1`, `cam0`, `cam1`は、`flat_active_grad0`, `flat_active_grad1`, `flat_active_feature` の値を元に算出したgradient x activationの値を250frame分になる様に補完をしたものである.

`flat_active_grad0`, `flat_active_grad1`, `flat_active_feature` は各Levelにおいて要素数が異なるが、その異なる要素数を1つのarrayに収めるために1次元の配列にflat化している. 例えば、kernel_size=2の場合、Level数は7になるが、acvtiveなgradient (有効なgradient)は各Levelにおいて `(63, 250), (63, 125), (63, 63), (63, 31), (63, 15), (63, 7), (63, 3)` となるが、それをflat化した1次元の配列 `(15750), (7875), (3969), (1953), (945), (441), (189)` として保持している.  `flat_active_*` は確認用で、実際の可視化にはそれらをもとに処理をした`cam_nopool*`や、`cam*`を利用することを想定している.



#### fMRI


| Key名 | 内容 |　Shape |
| ------------------ | ------------- | ------------- |
| cam0               | Global-Pooling利用, 最終層Grad-CAM, label=0に対する結果 | (*, 6, 7, 6)  |
| cam1               | Global-Pooling利用, 最終層Grad-CAM, label=1に対する結果 | (*, 6, 7, 6)  |
| cam_nopool0        | Global-Pooling利用しない, 最終層Grad-CAM, label=0に対する結果 | (*, 6, 7, 6) |
| cam_nopool1        | Global-Pooling利用しない, 最終層Grad-CAM, label=1に対する結果 | (*, 6, 7, 6) |
| guided_bp0         | Guided-Backpropの結果, label=0に対する結果 | (*, 79, 95, 79) |
| guided_bp1         | Guided-Backpropの結果, label=1に対する結果 | (*, 79, 95, 79) |
| guided_cam_nopool0 | Global-Pooling利用しない, Guided-GradCAMの結果, label=0に対する結果 | (*, 79, 95, 79) |
| guided_cam_nopool1 | Global-Pooling利用しない, Guided-GradCAMの結果, label=1に対する結果 | (*, 79, 95, 79) |
| label              | 正解ラベル (0 or 1) | (1, *) |
| predicted_label    | 予測ラベル (predicted_prob > 0.5なら1, そうでなければ0) | (1, *) |
| predicted_prob     | 予測確率 (0.0~1.0) | (1, *) |

`label`が1で`predicted_label`が1の場合は予測が正しかった場合であり、その時`guided_cam_nopool1が(正解である)ラベル1を予測した時のGuided-Grad-CAMの結果となる.
`

`label`が1で`predicted_label`が0の場合は、予測が外れたことを示す.



## 8. Grid search

グリッドサーチを行うには、`main_grid_search_example.py` を参考にしてグリッドサーチ用のスクリプトを用意して実行する.


スクリプトには以下の様に、実行ファイル、保存ディレクトリ名prefix, グリッドサーチ対象オプション、固定オプションを指定する.

```python
#-----[変更箇所 ここから]-----
# 実行するファイル名を指定
target_file = "main_eeg.py"

save_dir_prefix = "./saved_eeg"
# 保存ディレクトリは、"./saved_eeg_0_0", "./saved_eeg_0_1" ... などとなる.

# グリッドサーチ対象オプション
# 対処のオプションを配列で指定する
variable_options["lr"]           = ["0.1", "0.01", "0.001"]
variable_options["weight_decay"] = ["0.0", "0.01", "0.1"]

# 固定オプション
fixed_options["data_dir"]      = "./data2/DLD/Data_Converted"
fixed_options["model_type"]    = "filter2"
fixed_options["test_subjects"] = "TM_191008_01,TM_191009_01"
fixed_options["fold_size"]     = "1" # 1Foldのみを対象にして高速化する場合
#-----[変更箇所 ここまで]-----
```



サンプルと同様なスクリプトを用意して実行する.

```shell
python3 main_grid_search_sample.py
```



上記の例だと実行後に、`./saved_eeg_summary/summary.tx`に以下に様にサマリが出力される.
 (varidationのスコアが一番よかったものに**best**という表示がされる.)

```
[ClasslfyType0]
  [    ] ./saved_eeg_0_0: 50.0 (lr=0.1 weight_decay=0.0)
  [    ] ./saved_eeg_0_1: 50.0 (lr=0.1 weight_decay=0.01)
  [    ] ./saved_eeg_0_2: 50.0 (lr=0.1 weight_decay=0.1)
  [    ] ./saved_eeg_1_0: 50.0 (lr=0.01 weight_decay=0.0)
  [    ] ./saved_eeg_1_1: 50.0 (lr=0.01 weight_decay=0.01)
  [    ] ./saved_eeg_1_2: 50.0 (lr=0.01 weight_decay=0.1)
  [    ] ./saved_eeg_2_0: 50.0 (lr=0.001 weight_decay=0.0)
  [best] ./saved_eeg_2_1: 63.333333333333333 (lr=0.001 weight_decay=0.01)
  [    ] ./saved_eeg_2_2: 50.0 (lr=0.001 weight_decay=0.1)

[ClasslfyType1]
  [best] ./saved_eeg_0_0: 53.333333333333336 (lr=0.1 weight_decay=0.0)
  [    ] ./saved_eeg_0_1: 53.333333333333336 (lr=0.1 weight_decay=0.01)
  [    ] ./saved_eeg_0_2: 46.666666666666664 (lr=0.1 weight_decay=0.1)
...以下略  
```

