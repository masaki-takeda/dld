import os
import numpy as np


class VariableOptionNode:
    def __init__(self, key=None, value=None, index=None):
        self.key = key
        self.value = value
        self.index = index
        self.child_nodes = None

    def add_children(self, key, values):
        if self.child_nodes is None:
            self.child_nodes = [VariableOptionNode(key, value, i) for i, value in enumerate(values)]
        else:
            for child_node in self.child_nodes:
                child_node.add_children(key, values)

    def dump(self):
        if self.key is not None:
            print("{}:{}".format(self.key, self.value))

        if self.child_nodes is not None:
            for child_node in self.child_nodes:
                child_node.dump()

    def get_option_strings(self, for_summary=False):
        if self.key is None:
            option_string = ""
        else:
            if for_summary:
                option_string = "{}={}".format(self.key, self.value)
            else:
                option_string = "--{}={}".format(self.key, self.value)

        if self.child_nodes is None:
            return [option_string]

        ret = []
        for child_node in self.child_nodes:
            child_option_strings = child_node.get_option_strings(for_summary=for_summary)
            for child_option_string in child_option_strings:
                new_option_string = option_string + " " + child_option_string
                ret.append(new_option_string)
            
        return ret

    def get_suffix_strings(self):
        if self.index is None:
            suffix_string = ""
        else:
            suffix_string = "_{}".format(self.index)

        if self.child_nodes is None:
            return [suffix_string]

        ret = []
        for child_node in self.child_nodes:
            child_suffix_strings = child_node.get_suffix_strings()
            for child_suffix_string in child_suffix_strings:
                new_suffix_string = suffix_string + child_suffix_string
                ret.append(new_suffix_string)
            
        return ret


class GridSearch:
    def __init__(self,
                 target_file,
                 save_dir_prefix,
                 variable_options,
                 fixed_options):
        
        self.target_file      = target_file
        self.save_dir_prefix  = save_dir_prefix
        self.variable_options = variable_options
        self.fixed_options    = fixed_options

        self.root_node = VariableOptionNode()

        self.is_test = False
        if ("test" in self.fixed_options and self.fixed_options["test"] == "true"):
            # テスト評価時
            self.is_test = True
    
        for variable_option in list(variable_options.items()):
            key    = variable_option[0]
            values = variable_option[1]
            self.root_node.add_children(key, values)

            
    def get_command_lines(self):
        suffix_strings = self.root_node.get_suffix_strings()
        variable_option_strings = self.root_node.get_option_strings()
        
        fixed_option_string = ""
    
        for fixed_option in list(self.fixed_options.items()):
            key   = fixed_option[0]
            value = fixed_option[1]
            fixed_option_string += " --{}={}".format(key, value)
    
        command_lines = []
    
        for suffix_string, variable_option_string in zip(suffix_strings, variable_option_strings):
            # エラーをerror.txtに残すように
            command_line = "python3 {} --save_dir={}{}{}{} 3>&1 1>&2 2>&3 | tee -a {}{}/error.txt".format(
                self.target_file,
                self.save_dir_prefix,
                suffix_string,
                fixed_option_string,
                variable_option_string,
                self.save_dir_prefix,
                suffix_string,
            )
            command_lines.append(command_line)

        return command_lines

    
    def export_summary(self):
        suffix_strings = self.root_node.get_suffix_strings()
        
        directories = []
        
        results_ct0 = []
        results_ct1 = []
        results_ct2 = []
        
        for suffix_string in suffix_strings:
            directory = "{}{}".format(self.save_dir_prefix, suffix_string)
            directories.append(directory)

            if not self.is_test:
                file_path_ct0 = "{}{}/result_ct0.txt".format(self.save_dir_prefix, suffix_string)
                result = self.load_result(file_path_ct0)
                results_ct0.append(result)
            
                file_path_ct1 = "{}{}/result_ct1.txt".format(self.save_dir_prefix, suffix_string)
                result = self.load_result(file_path_ct1)
                results_ct1.append(result)
            
                file_path_ct2 = "{}{}/result_ct2.txt".format(self.save_dir_prefix, suffix_string)
                result = self.load_result(file_path_ct2)
                results_ct2.append(result)
            else:
                file_path_ct0 = "{}{}/result_ct0_test.txt".format(self.save_dir_prefix, suffix_string)
                result = self.load_result(file_path_ct0, for_test=True)
                results_ct0.append(result)
            
                file_path_ct1 = "{}{}/result_ct1_test.txt".format(self.save_dir_prefix, suffix_string)
                result = self.load_result(file_path_ct1, for_test=True)
                results_ct1.append(result)
            
                file_path_ct2 = "{}{}/result_ct2_test.txt".format(self.save_dir_prefix, suffix_string)
                result = self.load_result(file_path_ct2, for_test=True)
                results_ct2.append(result)
            
        max_index_ct0 = np.argmax(results_ct0)
        max_index_ct1 = np.argmax(results_ct1)
        max_index_ct2 = np.argmax(results_ct2)
        
        variable_option_strings = self.root_node.get_option_strings(for_summary=True)

        summary_lines = []

        summary_lines.append("[ClasslfyType0]")
        for i in range(len(directories)):
            if i == max_index_ct0:
                best = "[best]"
            else:
                best = "[    ]"
            summary_lines.append("  {} {}: {} ({})".format(best, directories[i], results_ct0[i],
                                                           variable_option_strings[i].strip()))
            
        summary_lines.append("")
        summary_lines.append("[ClasslfyType1]")
        for i in range(len(directories)):
            if i == max_index_ct1:
                best = "[best]"
            else:
                best = "[    ]"
            summary_lines.append("  {} {}: {} ({})".format(best, directories[i], results_ct1[i],
                                                           variable_option_strings[i].strip()))

        summary_lines.append("")
        summary_lines.append("[ClasslfyType2]")
        for i in range(len(directories)):
            if i == max_index_ct2:
                best = "[best]"
            else:
                best = "[    ]"
            summary_lines.append("  {} {}: {} ({})".format(best, directories[i], results_ct2[i],
                                                           variable_option_strings[i].strip()))

        summary_file_dir = self.get_summary_file_dir()
        if not os.path.exists(summary_file_dir):
            os.makedirs(summary_file_dir)

        summary_file_path = self.get_summary_file_path()
        f = open(summary_file_path, "w", encoding="utf_8")
        for line in summary_lines:
            f.write(line + "\n")
            print(line)
        f.close()

    def load_result(self, path, for_test=False):
        if not os.path.exists(path):
            return -np.inf
        
        with open(path) as f:
            lines = f.readlines()
        
        for line in lines:
            if not for_test:
                if "validation_accurcy_mean" in line:
                    return float(line.split('=')[1])
            else:
                if "test_accurcy_mean" in line:
                    return float(line.split('=')[1])
            
        return -np.inf

    def get_summary_file_dir(self):
        return "{}{}".format(self.save_dir_prefix, "_summary")

    def get_summary_file_path(self):
        if not self.is_test:
            return "{}/{}".format(self.get_summary_file_dir(), "summary.txt")
        else:
            return "{}/{}".format(self.get_summary_file_dir(), "summary_test.txt")
