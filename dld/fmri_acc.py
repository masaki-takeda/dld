import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 数据文件路径
data_dirs = {
    9: '/data3/DLD2/241210_eeg/a9r9/class4',
    7: '/data3/DLD2/241210_eeg/a7r7/class4',
    5: '/data3/DLD2/241210_eeg/a5r5/class4',
    3: '/data3/DLD2/241210_eeg/a3r3/class4',
    1: '/data3/DLD2/241210_eeg/a1r1/class4'
}
# 模型数量（0 到 8，共 9 个模型）
models = 9

num_trials = np.array([1, 3, 5, 7, 9])  # 平均试验数

# 存储每个试验次数的准确率
accuracies = []
average_accuracies = []  # 每个试验次数的平均准确率
std_errors = []  # 每个试验次数的标准误差

# 遍历不同的试验次数目录，读取准确率数据
for trial in num_trials:
    data_dir = data_dirs[trial]
    
    # 获取每个模型的准确率数据
    raw_test_file = os.path.join(data_dir, 'result_ct4_raw_test.csv')
    df_raw = pd.read_csv(raw_test_file, index_col=0)  # 设置索引为模型编号
    
    # 提取准确率数据
    accuracy = df_raw.iloc[:, 0].values  # 第一列是模型准确率 
    # 将准确率添加到列表中
    accuracies.append(accuracy)

    # 从 result_ct4_test.csv 中提取平均准确率和标准误
    test_file = os.path.join(data_dir, 'result_ct4_test.csv')
    df_test = pd.read_csv(test_file, index_col=0)
    
    average_accuracies.append(df_test.iloc[0, 0])  # 第一列的第 1 行是平均值
    std_errors.append(df_test.iloc[1, 0])  # 第一列的第 2 行是标准误差

# 转换为 NumPy 数组 (5 个试验次数, 每个试验 9 个模型)
accuracies = np.array(accuracies)
average_accuracies = np.array(average_accuracies)  # 平均准确率
std_errors = np.array(std_errors)  # 标准误差

# 创建绘图
plt.figure(figsize=(15, 9))

# 每个模型的准确率曲线 (灰色)
for model in range(models):
    plt.plot(num_trials, accuracies[:, model], '-o', color='gray', alpha=1.0, label='Each model' if model == 0 else "")

# 平均值曲线和误差条 (黑色)
plt.errorbar(num_trials, average_accuracies, yerr=std_errors, fmt='-o', color='black', label='Average', linewidth=1.5)

# 集成学习性能曲线 (紫色)
# plt.plot(num_trials, ensemble_learning, '-o', color='magenta', label='Ensemble learning', linewidth=2)

# 添加图例、标签和标题
plt.legend()
plt.xlabel('Number of averaged trials', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('big vs small', fontsize=14)

# 设置纵坐标范围
plt.ylim(45, 75)  # 根据数据范围手动设置
plt.grid(alpha=0.0)

# 保存图像
plt.tight_layout()
plt.savefig('/data3/DLD2/241210_eeg/big_vs_small_eeg.png')  # 保存为PNG文件

# 显示绘图
plt.show()
