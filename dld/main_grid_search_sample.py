# Template of grid search execution
from collections import OrderedDict
import subprocess
from subprocess import Popen
from grid_search import GridSearch

variable_options = OrderedDict()
fixed_options = OrderedDict()

#-----[Changes from here]-----
# Specify the name of the file to execute
target_file = "main_eeg.py"

save_dir_prefix = "./saved_eeg"
# The save directory should be "./saved_eeg_0_0", "./saved_eeg_0_1" ... and so on

#save_dir_prefix = "./saved_eeg/gs"
# When specifying up to a subdirectory as above
# The save directory will be "./saved_eeg/gs_0_0", "./saved_eeg/gs_0_1" ... and so on

# Grid search target options. Specify target options as an array
variable_options["lr"]           = ["0.1", "0.01", "0.001"]
variable_options["weight_decay"] = ["0.0", "0.01", "0.1"]


# Fixed options
fixed_options["data_dir"]      = "./data2/DLD/Data_Converted"
fixed_options["model_type"]    = "filter2"
fixed_options["test_subjects"] = "TM_191008_01,TM_191009_01"
fixed_options["fold_size"]     = "1" # When only 1 Fold is targeted to speed up the processing
fixed_options["gpu"]           = "1"
#-----[To here]-----


grid_search = GridSearch(target_file,
                         save_dir_prefix,
                         variable_options,
                         fixed_options)

command_lines = grid_search.get_command_lines()

# Grid search execution
for command_line in command_lines:
    print("executing: {}".format(command_line))
    proc = Popen(command_line, shell=True)
    proc.wait()

# Output the summarized results of grid search to "saved_eeg00_summary/summary.txt"
grid_search.export_summary()
