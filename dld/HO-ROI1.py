import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from nilearn.image import resample_to_img
import nibabel as nib
from statsmodels.stats.multitest import fdrcorrection
import os
from nilearn.datasets import fetch_atlas_harvard_oxford


# 1. 加载Guided Grad-CAM数据
def load_gradcam_data(learned_data_path):
    learned_data = np.load(learned_data_path)

    label_learned = learned_data['label'].flatten()  # (1, *)
    predicted_label_learned = learned_data['predicted_label'].flatten()
    guided_cam_nopool0_learned = learned_data['guided_cam_nopool0']  # (782, 79, 95, 79)
    guided_cam_nopool1_learned = learned_data['guided_cam_nopool1']

    ct_data_learned = {}

    ct_data_learned['Success_trial'] = {
        'type0': np.where((label_learned == 0) & (predicted_label_learned == 0))[0],
        'type1': np.where((label_learned == 1) & (predicted_label_learned == 1))[0]
    }
    ct_data_learned['Error_trial'] = {
        'type0': np.where((label_learned == 0) & (predicted_label_learned == 1))[0],
        'type1': np.where((label_learned == 1) & (predicted_label_learned == 0))[0]
    }

    ct_data_learned['guided_cam_nopool'] = {
        'type0': {  # 对于标签0（小）
            'Success': guided_cam_nopool0_learned[ct_data_learned['Success_trial']['type0']],
            'Error': guided_cam_nopool0_learned[ct_data_learned['Error_trial']['type0']]
        },
        'type1': {  # 对于标签1（大）
            'Success': guided_cam_nopool1_learned[ct_data_learned['Success_trial']['type1']],
            'Error': guided_cam_nopool1_learned[ct_data_learned['Error_trial']['type1']]
        }
    }

    return ct_data_learned

def map_region_to_category(region_name):
    if any(area in region_name for area in ['Temporal', 'MTG', 'STG', 'ITG', 'TP', 'Heschl', 'TPO']):
        return 'Temporal'

    # 枕叶区域
    elif any(area in region_name for area in ['Occipital', 'LOC', 'TOF', 'Visual', 'Lingual', 'Cuneal', 'Calcarine']):
        return 'Occipital'

    # 额叶区域
    elif any(area in region_name for area in ['Frontal', 'IFG', 'MFG', 'SFG', 'OFC', 'Precentral']):
        return 'Frontal'

    # 顶叶区域
    elif any(area in region_name for area in ['Parietal', 'SPL', 'IPL', 'SMG', 'Angular', 'Postcentral']):
        return 'Parietal'

    # 边缘系统
    elif any(area in region_name for area in ['Cingulate', 'Paracingulate', 'Insular', 'Insula']):
        return 'Limbic'

    # 基底核区域
    elif any(area in region_name for area in ['Putamen', 'Caudate', 'Thalamus', 'Pallidum', 'Accumbens', 'Amygdala']):
        return 'Subcortical'

    # 小脑
    elif 'Cerebellum' in region_name:
        return 'Cerebellum'

    # 脑干
    elif 'Brain-Stem' in region_name:
        return 'Brain-Stem'

    # 其他皮层区域
    elif 'Cortical' in region_name:
        return 'Cortical-Other'

    # 默认返回原类型
    else:
        return 'Other'

# 2. 使用nilearn加载哈佛-牛津图谱
def load_harvard_oxford_atlas():
    ho_cortical = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
    ho_subcortical = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm', symmetric_split=True)

    cortical_data = ho_cortical['maps'].get_fdata()
    subcortical_data = ho_subcortical['maps'].get_fdata()

    cortical_labels = ho_cortical['labels']
    subcortical_labels = ho_subcortical['labels']

    # 合并皮层和皮层下图谱
    # 皮层标签索引从0开始，皮层下标签需要偏移
    max_cortical_id = np.max(cortical_data)
    subcortical_data[subcortical_data > 0] += max_cortical_id

    # 创建合并的图谱
    # 对于重叠区域，优先保留皮层下结构
    atlas_data = cortical_data.copy()
    atlas_data[subcortical_data > 0] = subcortical_data[subcortical_data > 0]

    # 处理标签以适应左右半球格式
    cortical_labels = cortical_labels[1:]  # 跳过背景标签
    subcortical_labels = subcortical_labels[1:]  # 跳过背景标签

    # 创建格式化的标签列表
    formatted_labels = []
    region_types = []
    detailed_region_types = []

    # 处理皮层标签
    for label in cortical_labels:
        if 'Left' in label:
            # 替换并格式化左侧标签
            region_name = label.replace('Left ', '')
            formatted_name = f"L. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        elif 'Right' in label:
            # 替换并格式化右侧标签
            region_name = label.replace('Right ', '')
            formatted_name = f"R. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        else:
            # 处理不包含左右信息的标签
            formatted_labels.append(label)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(label))

    # 处理皮层下标签
    for label in subcortical_labels:
        if 'Left' in label:
            # 替换并格式化左侧标签
            region_name = label.replace('Left ', '')
            formatted_name = f"L. {region_name}"  # 注意：使用 L. 保持一致
            formatted_labels.append(formatted_name)
            region_types.append('Subcortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        elif 'Right' in label:
            # 替换并格式化右侧标签
            region_name = label.replace('Right ', '')
            formatted_name = f"R. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Subcortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        else:
            # 处理不包含左右信息的标签
            formatted_labels.append(label)
            region_types.append('Subcortical')
            detailed_region_types.append(map_region_to_category(label))

    # 创建标签DataFrame
    region_labels = pd.DataFrame({
        'Region_ID': list(range(1, len(formatted_labels) + 1)),
        'Region': formatted_labels,
        'Type': region_types,
        'Detailed_Type': detailed_region_types
    })

    print(f"哈佛-牛津图谱共加载了 {len(region_labels)} 个解剖区域")
    return atlas_data, region_labels

# 4. 计算每个ROI的Guided Grad-CAM平均值
def calculate_roi_means(gradcam_data, atlas_data, region_labels):
    ho_cortical = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = ho_cortical['maps']
    atlas_affine = atlas_img.affine

    num_regions = len(region_labels)
    num_samples = gradcam_data.shape[0]
    roi_means = np.zeros((num_samples, num_regions))

    for sample_idx in range(num_samples):
        sample_gradcam = gradcam_data[sample_idx]  # shape: (79, 95, 79)

        gradcam_img = nib.Nifti1Image(sample_gradcam, affine=atlas_affine)

        resampled_gradcam_img = resample_to_img(gradcam_img, atlas_img, interpolation='linear')
        resampled_gradcam = resampled_gradcam_img.get_fdata()

        # 对每个区域计算平均值
        for region_idx, region_row in region_labels.iterrows():
            region_id = region_row['Region_ID']
            region_mask = (atlas_data == region_id)

            if np.sum(region_mask) == 0:
                continue

            # 计算该区域的平均值
            region_values = resampled_gradcam[region_mask]
            if len(region_values) > 0:
                roi_means[sample_idx, region_idx] = np.mean(region_values)

    print(f"重采样和ROI平均值计算完成")
    return roi_means

# 5. 与0进行统计比较和FDR校正
def statistical_comparison_with_zero(learned_roi_means, region_labels, alpha=0.05):
    num_regions = learned_roi_means.shape[1]
    p_values = np.zeros(num_regions)
    effect_sizes = np.zeros(num_regions)

    # 对每个区域进行Wilcoxon符号秩检验（与0比较）
    for region_idx in range(num_regions):
        learned_values = learned_roi_means[:, region_idx]

        # 如果存在非零值，则进行Wilcoxon符号秩检验
        if np.any(learned_values != 0):
            # Wilcoxon符号秩检验（与0比较）
            stat, p_value = wilcoxon(learned_values, alternative='two-sided')
            p_values[region_idx] = p_value
        else:
            p_values[region_idx] = 1.0

        # 计算效应量（平均值）
        effect_sizes[region_idx] = np.mean(learned_values)

    # FDR校正 (Benjamini-Hochberg方法)
    reject, q_values = fdrcorrection(p_values, alpha=alpha, method='indep')

    # 准备结果表格
    results = []
    for region_idx in range(num_regions):
        region_name = region_labels.iloc[region_idx]['Region']
        region_type = region_labels.iloc[region_idx]['Type']
        detailed_type = region_labels.iloc[region_idx]['Detailed_Type']
        region_id = region_labels.iloc[region_idx]['Region_ID']

        # 计算平均值
        learned_mean = np.mean(learned_roi_means[:, region_idx])
        effect = effect_sizes[region_idx]

        # 添加显著性标记
        significance = ''
        if reject[region_idx]:
            significance = '*'

        results.append({
            'Region_ID': region_id,
            'Region_Name': region_name,
            'Region_Type': region_type,
            'Detailed_Type': detailed_type,
            'Learned_Mean': learned_mean,  # 学习组的guided grad-cam平均值
            'Effect_Size': effect,
            'P_Value': p_values[region_idx],
            'FDR_Q_Value': q_values[region_idx],
            'Significant': reject[region_idx],
            'Significance': significance
        })

    results_df = pd.DataFrame(results)
    return results_df

# 6. 可视化和输出结果
def results(results_df, output_dir, data_type):
    os.makedirs(output_dir, exist_ok=True)

    # 重命名列并选择需要的列
    output_table = results_df[['Region_Name', 'Detailed_Type', 'Learned_Mean', 'FDR_Q_Value', 'Significance']]
    output_table = output_table.rename(columns={
        'Region_Name': 'Label',
        'Detailed_Type': 'Region',          # 详细类型（Temporal/Occipital等）
        'Learned_Mean': 'Value',            # 显示学习组的guided grad-cam平均值
        'FDR_Q_Value': 'FDR q',
        'Significance': 'Significance'
    })

    # 保存到CSV
    output_table.to_csv(os.path.join(output_dir, f'gradcam_comparison_results_{data_type}.csv'), index=False)

    print(f"{data_type} 结果表格已保存至 {output_dir}")

# 主函数
def main():
    # 文件路径配置（根据实际情况修改）
    learned_data_path = '/data3/DLD2/241212_fmri/a9r9/class4/grad_cam/data/cam_fmri_ct4_3.npz'
    output_dir = '/data3/DLD2/241212_fmri/results_HO1'

    learned_data = load_gradcam_data(learned_data_path)

    atlas_data, region_labels = load_harvard_oxford_atlas()

    for data_type in ['type0', 'type1']:
        print(f"\n====== 处理 {data_type} 数据 ======")

        print(f"计算 {data_type} 各脑区ROI的Guided Grad-CAM平均值...")
        learned_roi_means = calculate_roi_means(learned_data['guided_cam_nopool'][data_type]['Success'], atlas_data, region_labels)

        print(f"{data_type} 进行统计比较和FDR校正...")
        results_df = statistical_comparison_with_zero(learned_roi_means, region_labels)

        significant_regions = results_df[results_df['Significant']]
        if len(significant_regions) > 0:
            display_cols = ['Region_Name', 'Region_Type', 'Effect_Size', 'FDR_Q_Value', 'Significance']
            print(significant_regions[display_cols].sort_values('FDR_Q_Value'))
        else:
            print(f"{data_type}: 未发现显著差异的脑区")

        results(results_df, output_dir, data_type)

    print("\n=== 脑区分析完成 ===")


if __name__ == "__main__":
    main()
