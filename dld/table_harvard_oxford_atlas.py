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
    guided_cam_nopool0_learned = learned_data['f_guided_cam_nopool0']  # (782, 79, 95, 79)
    guided_cam_nopool1_learned = learned_data['f_guided_cam_nopool1']

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
    if any(area in region_name for area in ['Temporal', 'Planum Polare', 'Parahippocampal Gyrus', 'MTG', 'STG', 'ITG', 'TP', 'Heschl', 'TPO']):
        return 'Temporal'

    # 枕叶区域
    elif any(area in region_name for area in ['Occipital', 'Intracalcarine Cortex', 'Supracalcarine Cortex', 'LOC', 'TOF', 'Visual', 'Lingual', 'Cuneal', 'Calcarine']):
        return 'Occipital'

    # 额叶区域
    elif any(area in region_name for area in ['Frontal', 'Juxtapositional Lobule Cortex', 'IFG', 'MFG', 'SFG', 'OFC', 'Precentral']):
        return 'Frontal'

    # 顶叶区域
    elif any(area in region_name for area in ['Parietal', 'Supramarginal Gyrus', 'Precuneous Cortex', 'SPL', 'IPL', 'SMG', 'Angular', 'Postcentral']):
        return 'Parietal'

    # 边缘系统
    elif any(area in region_name for area in ['Cingulate', 'Paracingulate', 'Insular', 'Insula']):
        return 'Limbic'

    # 基底核区域
    elif any(area in region_name for area in ['Hippocampus', 'Lateral Ventricle', 'Putamen', 'Caudate', 'Thalamus', 'Pallidum', 'Accumbens', 'Amygdala']):
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

    cortical_img = ho_cortical['maps']
    subcortical_img = ho_subcortical['maps']

    cortical_labels = ho_cortical['labels']
    subcortical_labels = ho_subcortical['labels']

    cortical_labels = cortical_labels[1:]  # 跳过背景标签
    subcortical_labels = subcortical_labels[1:]  # 跳过背景标签

    formatted_labels = []
    region_types = []
    detailed_region_types = []

    # 处理皮层标签
    for label in cortical_labels:
        if 'Left' in label:
            region_name = label.replace('Left ', '')
            formatted_name = f"L. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        elif 'Right' in label:
            region_name = label.replace('Right ', '')
            formatted_name = f"R. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        else:
            formatted_labels.append(label)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(label))

    # 处理皮层下标签，并调整ID以避免与皮层标签重叠
    max_cortical_id = len(cortical_labels)

    for i, label in enumerate(subcortical_labels):
        if 'Left' in label:
            region_name = label.replace('Left ', '')
            formatted_name = f"L. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Subcortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        elif 'Right' in label:
            region_name = label.replace('Right ', '')
            formatted_name = f"R. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Subcortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        else:
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

    # 返回原始nifti图像和标签
    return {
        'cortical_img': cortical_img,
        'subcortical_img': subcortical_img,
        'region_labels': region_labels,
        'max_cortical_id': max_cortical_id
    }

# 4. 计算每个ROI的Guided Grad-CAM平均值
def calculate_roi_means(gradcam_data, atlas_info):
    # 使用原始的atlas图像进行重采样
    cortical_img = atlas_info['cortical_img']
    subcortical_img = atlas_info['subcortical_img']
    region_labels = atlas_info['region_labels']
    max_cortical_id = atlas_info['max_cortical_id']

    cortical_data = cortical_img.get_fdata()
    subcortical_data = subcortical_img.get_fdata()

    # 从atlas_info中获取subcortical_labels
    subcortical_labels = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm', symmetric_split=True)['labels'][1:]

    num_regions = len(region_labels)
    num_samples = gradcam_data.shape[0]
    roi_means = np.zeros((num_samples, num_regions))

    reference_affine = cortical_img.affine

    for sample_idx in range(num_samples):
        sample_gradcam = gradcam_data[sample_idx]  # shape: (79, 95, 79)

        # 创建gradcam的nifti图像对象，使用图谱的仿射矩阵
        gradcam_img = nib.Nifti1Image(sample_gradcam, affine=reference_affine)

        # 分别处理皮层和皮层下区域
        # 1. 皮层区域
        for region_idx in range(max_cortical_id):
            region_id = region_idx + 1  # 区域ID从1开始

            # 创建单个区域的掩码图像
            region_mask = np.zeros_like(cortical_data)
            region_mask[cortical_data == region_id] = 1
            region_mask_img = nib.Nifti1Image(region_mask, cortical_img.affine)

            # 将区域掩码重采样到gradcam空间
            resampled_mask = resample_to_img(
                region_mask_img,
                gradcam_img,
                interpolation='nearest'
            ).get_fdata()

            # 提取该区域的gradcam值
            if np.sum(resampled_mask) > 0:
                roi_mean = np.mean(sample_gradcam[resampled_mask > 0.5])
                roi_means[sample_idx, region_idx] = roi_mean

        # 2. 皮层下区域
        for sub_idx in range(len(subcortical_labels)):  # 使用获取的subcortical_labels
            region_idx = max_cortical_id + sub_idx
            sub_id = sub_idx + 1  # 皮层下区域ID从1开始

            # 创建单个区域的掩码图像
            region_mask = np.zeros_like(subcortical_data)
            region_mask[subcortical_data == sub_id] = 1
            region_mask_img = nib.Nifti1Image(region_mask, subcortical_img.affine)

            # 将区域掩码重采样到gradcam空间
            resampled_mask = resample_to_img(
                region_mask_img,
                gradcam_img,
                interpolation='nearest'
            ).get_fdata()

            # 提取该区域的gradcam值
            if np.sum(resampled_mask) > 0:
                roi_mean = np.mean(sample_gradcam[resampled_mask > 0.5])
                roi_means[sample_idx, region_idx] = roi_mean

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
    output_table.to_csv(os.path.join(output_dir, f'gradcam_combined_comparison_results_{data_type}.csv'), index=False)

    print(f"{data_type} 结果表格已保存至 {output_dir}")

# 主函数
def main():
    # 文件路径配置（根据实际情况修改）
    learned_data_path = '/data3/DLD2/241225_combined/a7r7/class4/grad_cam/data/cam_combined_ct4_2.npz'
    output_dir = '/data3/DLD2/241225_combined/results_HO'

    learned_data = load_gradcam_data(learned_data_path)

    atlas_info = load_harvard_oxford_atlas()
    region_labels = atlas_info['region_labels']

    for data_type in ['type0', 'type1']:
        print(f"\n====== 处理 {data_type} 数据 ======")

        print(f"计算 {data_type} 各脑区ROI的Guided Grad-CAM平均值...")
        learned_roi_means = calculate_roi_means(learned_data['guided_cam_nopool'][data_type]['Success'], atlas_info)

        print(f"{data_type} 进行统计比较和FDR校正...")
        results_df = statistical_comparison_with_zero(learned_roi_means, region_labels)

        significant_regions = results_df[results_df['Significant']]
        if len(significant_regions) > 0:
            display_cols = ['Region_Name', 'Region_Type', 'Effect_Size', 'FDR_Q_Value', 'Significance']
            print(significant_regions[display_cols].sort_values('FDR_Q_Value'))
        else:
            print(f"{data_type}: 未发现显著差异的脑区")

        results(results_df, output_dir, data_type)


if __name__ == "__main__":
    main()
