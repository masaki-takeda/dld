# グリッドサーチ実行の雛形
from collections import OrderedDict
import subprocess
from subprocess import Popen
from grid_search import GridSearch

variable_options = OrderedDict()
fixed_options = OrderedDict()

#-----[変更箇所 ここから]-----
# 実行するファイル名を指定
target_file = "main_eeg.py"

save_dir_prefix = "./saved_eeg"
# 保存ディレクトリは、"./saved_eeg_0_0", "./saved_eeg_0_1" ... などとなる.

#save_dir_prefix = "./saved_eeg/gs"
#などとサブディレクトリまでを指定すると
# "./saved_eeg/gs_0_0", "./saved_eeg/gs_0_1" ... などディレクトリが作られる.

# グリッドサーチ対象オプション
# 対処のオプションを配列で指定する
variable_options["lr"]           = ["0.1", "0.01", "0.001"]
variable_options["weight_decay"] = ["0.0", "0.01", "0.1"]


# 固定オプション
fixed_options["data_dir"]      = "./data2/DLD/Data_Converted"
fixed_options["model_type"]    = "filter2"
fixed_options["test_subjects"] = "TM_191008_01,TM_191009_01"
fixed_options["fold_size"]     = "1" # 1Foldのみを対象にして高速化する場合
fixed_options["gpu"]           = "1"
#-----[変更箇所 ここまで]-----


grid_search = GridSearch(target_file,
                         save_dir_prefix,
                         variable_options,
                         fixed_options)

command_lines = grid_search.get_command_lines()

# グリッドサーチ実行
for command_line in command_lines:
    print("executing: {}".format(command_line))
    proc = Popen(command_line, shell=True)
    proc.wait()

# グリッドサーチ結果サマリ出力
# saved_eeg00_summary/summary.txt にサマリを出力する.
grid_search.export_summary()
