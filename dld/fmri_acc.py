import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_dirs = {
    9: '/data3/DLD2/241210_eeg/a9r9/class4',
    7: '/data3/DLD2/241210_eeg/a7r7/class4',
    5: '/data3/DLD2/241210_eeg/a5r5/class4',
    3: '/data3/DLD2/241210_eeg/a3r3/class4',
    1: '/data3/DLD2/241210_eeg/a1r1/class4'
}
# Number of models
models = 9

num_trials = np.array([1, 3, 5, 7, 9])  # Number of average trials

# Store the accuracy rate for each trial
accuracies = []
average_accuracies = [] 
std_errors = [] 

# Iterate through the directories containing different test runs and read the accuracy data
for trial in num_trials:
    data_dir = data_dirs[trial]
    
    # Retrieve accuracy data for each model
    raw_test_file = os.path.join(data_dir, 'result_ct4_raw_test.csv')
    df_raw = pd.read_csv(raw_test_file, index_col=0)

    accuracy = df_raw.iloc[:, 0].values
    accuracies.append(accuracy)

    test_file = os.path.join(data_dir, 'result_ct4_test.csv')
    df_test = pd.read_csv(test_file, index_col=0)
    
    average_accuracies.append(df_test.iloc[0, 0]) 
    std_errors.append(df_test.iloc[1, 0])

# Convert to a NumPy array
accuracies = np.array(accuracies)
average_accuracies = np.array(average_accuracies)
std_errors = np.array(std_errors)

plt.figure(figsize=(15, 9))

for model in range(models):
    plt.plot(num_trials, accuracies[:, model], '-o', color='gray', alpha=1.0, label='Each model' if model == 0 else "")

plt.errorbar(num_trials, average_accuracies, yerr=std_errors, fmt='-o', color='black', label='Average', linewidth=1.5)


plt.legend()
plt.xlabel('Number of averaged trials', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('big vs small', fontsize=14)

plt.ylim(45, 75) 
plt.grid(alpha=0.0)

plt.tight_layout()
plt.savefig('/data3/DLD2/241210_eeg/big_vs_small_eeg.png')

plt.show()
