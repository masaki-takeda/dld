import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from nilearn.image import resample_to_img
import nibabel as nib
from statsmodels.stats.multitest import fdrcorrection
import os
from nilearn.datasets import fetch_atlas_harvard_oxford


# 1. Load Guided Grad-CAM data
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
        'type0': {  # small
            'Success': guided_cam_nopool0_learned[ct_data_learned['Success_trial']['type0']],
            'Error': guided_cam_nopool0_learned[ct_data_learned['Error_trial']['type0']]
        },
        'type1': {  # large
            'Success': guided_cam_nopool1_learned[ct_data_learned['Success_trial']['type1']],
            'Error': guided_cam_nopool1_learned[ct_data_learned['Error_trial']['type1']]
        }
    }

    return ct_data_learned

def map_region_to_category(region_name):
    if any(area in region_name for area in ['Temporal', 'MTG', 'STG', 'ITG', 'TP', 'Heschl', 'TPO']):
        return 'Temporal'

    elif any(area in region_name for area in ['Occipital', 'LOC', 'TOF', 'Visual', 'Lingual', 'Cuneal', 'Calcarine']):
        return 'Occipital'

    elif any(area in region_name for area in ['Frontal', 'IFG', 'MFG', 'SFG', 'OFC', 'Precentral']):
        return 'Frontal'

    elif any(area in region_name for area in ['Parietal', 'SPL', 'IPL', 'SMG', 'Angular', 'Postcentral']):
        return 'Parietal'

    elif any(area in region_name for area in ['Cingulate', 'Paracingulate', 'Insular', 'Insula']):
        return 'Limbic'

    elif any(area in region_name for area in ['Putamen', 'Caudate', 'Thalamus', 'Pallidum', 'Accumbens', 'Amygdala']):
        return 'Subcortical'

    elif 'Cerebellum' in region_name:
        return 'Cerebellum'

    elif 'Brain-Stem' in region_name:
        return 'Brain-Stem'

    elif 'Cortical' in region_name:
        return 'Cortical-Other'

    else:
        return 'Other'

# 2. Loading the Harvard-Oxford Atlas using nilearn
def load_harvard_oxford_atlas():
    ho_cortical = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
    ho_subcortical = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm', symmetric_split=True)

    cortical_data = ho_cortical['maps'].get_fdata()
    subcortical_data = ho_subcortical['maps'].get_fdata()

    cortical_labels = ho_cortical['labels']
    subcortical_labels = ho_subcortical['labels']

    # Merge cortical and subcortical atlases
    max_cortical_id = np.max(cortical_data)
    subcortical_data[subcortical_data > 0] += max_cortical_id

    # Create a merged graph
    atlas_data = cortical_data.copy()
    atlas_data[subcortical_data > 0] = subcortical_data[subcortical_data > 0]

    # Process tags to accommodate left-right layout
    cortical_labels = cortical_labels[1:] 
    subcortical_labels = subcortical_labels[1:] 

    formatted_labels = []
    region_types = []
    detailed_region_types = []

    # Handling layer labels
    for label in cortical_labels:
        if 'Left' in label:
            # Replace and format the left-hand labels
            region_name = label.replace('Left ', '')
            formatted_name = f"L. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        elif 'Right' in label:
            # Replace and format the label on the right
            region_name = label.replace('Right ', '')
            formatted_name = f"R. {region_name}"
            formatted_labels.append(formatted_name)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(formatted_name))
        else:
            # Handling tags that do not include left and right information
            formatted_labels.append(label)
            region_types.append('Cortical')
            detailed_region_types.append(map_region_to_category(label))

    # Handling Subcortical Labels
    for label in subcortical_labels:
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

    region_labels = pd.DataFrame({
        'Region_ID': list(range(1, len(formatted_labels) + 1)),
        'Region': formatted_labels,
        'Type': region_types,
        'Detailed_Type': detailed_region_types
    })

    print(f"The Harvard-Oxford Atlas has loaded a total of {len(region_labels)} anatomical regions.")
    return atlas_data, region_labels

# 4. Calculate the average Guided Grad-CAM score for each ROI
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

        for region_idx, region_row in region_labels.iterrows():
            region_id = region_row['Region_ID']
            region_mask = (atlas_data == region_id)

            if np.sum(region_mask) == 0:
                continue

            region_values = resampled_gradcam[region_mask]
            if len(region_values) > 0:
                roi_means[sample_idx, region_idx] = np.mean(region_values)

    print(f"Resampling and ROI averaging are complete")
    return roi_means

# 5. Statistical comparison with zero and FDR correction
def statistical_comparison_with_zero(learned_roi_means, region_labels, alpha=0.05):
    num_regions = learned_roi_means.shape[1]
    p_values = np.zeros(num_regions)
    effect_sizes = np.zeros(num_regions)

    # Perform a Wilcoxon signed-rank test on each region (compared to 0)
    for region_idx in range(num_regions):
        learned_values = learned_roi_means[:, region_idx]

        if np.any(learned_values != 0):
            stat, p_value = wilcoxon(learned_values, alternative='two-sided')
            p_values[region_idx] = p_value
        else:
            p_values[region_idx] = 1.0

        effect_sizes[region_idx] = np.mean(learned_values)

    # FDR Correction (Benjamini-Hochberg Method)
    reject, q_values = fdrcorrection(p_values, alpha=alpha, method='indep')

    results = []
    for region_idx in range(num_regions):
        region_name = region_labels.iloc[region_idx]['Region']
        region_type = region_labels.iloc[region_idx]['Type']
        detailed_type = region_labels.iloc[region_idx]['Detailed_Type']
        region_id = region_labels.iloc[region_idx]['Region_ID']

        learned_mean = np.mean(learned_roi_means[:, region_idx])
        effect = effect_sizes[region_idx]

        significance = ''
        if reject[region_idx]:
            significance = '*'

        results.append({
            'Region_ID': region_id,
            'Region_Name': region_name,
            'Region_Type': region_type,
            'Detailed_Type': detailed_type,
            'Learned_Mean': learned_mean,
            'Effect_Size': effect,
            'P_Value': p_values[region_idx],
            'FDR_Q_Value': q_values[region_idx],
            'Significant': reject[region_idx],
            'Significance': significance
        })

    results_df = pd.DataFrame(results)
    return results_df

# 6. Visualization and Output Results
def results(results_df, output_dir, data_type):
    os.makedirs(output_dir, exist_ok=True)

    output_table = results_df[['Region_Name', 'Detailed_Type', 'Learned_Mean', 'FDR_Q_Value', 'Significance']]
    output_table = output_table.rename(columns={
        'Region_Name': 'Label',
        'Detailed_Type': 'Region',  
        'Learned_Mean': 'Value',      
        'FDR_Q_Value': 'FDR q',
        'Significance': 'Significance'
    })

    output_table.to_csv(os.path.join(output_dir, f'gradcam_comparison_results_{data_type}.csv'), index=False)

    print(f"The {data_type} results table has been saved to {output_dir}")

def main():
    learned_data_path = '/data3/DLD2/241212_fmri/a9r9/class4/grad_cam/data/cam_fmri_ct4_3.npz'
    output_dir = '/data3/DLD2/241212_fmri/results_HO1'

    learned_data = load_gradcam_data(learned_data_path)

    atlas_data, region_labels = load_harvard_oxford_atlas()

    for data_type in ['type0', 'type1']:
        print(f"\n====== Processing {data_type} Data ======")

        print(f"Calculate the average Guided Grad-CAM scores for the {data_type} ROIs in each brain region...")
        learned_roi_means = calculate_roi_means(learned_data['guided_cam_nopool'][data_type]['Success'], atlas_data, region_labels)

        print(f"Perform statistical comparisons and FDR correction on {data_type}...")
        results_df = statistical_comparison_with_zero(learned_roi_means, region_labels)

        significant_regions = results_df[results_df['Significant']]
        if len(significant_regions) > 0:
            display_cols = ['Region_Name', 'Region_Type', 'Effect_Size', 'FDR_Q_Value', 'Significance']
            print(significant_regions[display_cols].sort_values('FDR_Q_Value'))
        else:
            print(f"{data_type}: Brain regions where no significant differences were found")

        results(results_df, output_dir, data_type)

    print("\n=== Brain Region Analysis Complete ===")


if __name__ == "__main__":
    main()
