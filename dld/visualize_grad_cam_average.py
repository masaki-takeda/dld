import numpy as np
import argparse
import os    
import matplotlib.pyplot as plt

from dataset import FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL

def export_fcam(fcam, title, file_path):
    fig, axes = plt.subplots(6, 1, figsize=(4.8+0.5, 6.4*4), tight_layout=True)
    
    vmax = np.max(fcam)
    vmin = 0.0

    for i in range(6):
        ax = axes[i]
        if i == 0:
            ax.set_title(title)
        im = ax.imshow(fcam[i],
                       interpolation='bicubic',
                       vmin=vmin,
                       vmax=vmax,
               cmap='jet')
    
    axpos = axes[0].get_position()
    cbar_ax = fig.add_axes([0.87, axpos.y0 + 0.105 , 0.02, axpos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(file_path)
    plt.close()


def aggregate_grad_cam(output_dir, classify_type, type):
    mean_cam_nopools_label0 = []
    mean_cam_nopools_label1 = []
    
    cam_nopools_label0 = []
    cam_nopools_label1 = []
    
    for i in range(10):
        data = np.load("{}/data/grad_cam_data_{}_ct{}_{}.npz".format(
            output_dir,
            type, classify_type, i),
                       allow_pickle=True)
        cam_nopool = data['cam_nopool'] # (1807, 3, 28)

        if type == 'eeg':
            cam_nopool = np.mean(cam_nopool, axis=1) # (1807, 28) Take the average of the 3 channels
    
        labels = data['label'] # (1807,)
        
        cam_nopool_label0 = cam_nopool[(labels==0)] # (910, 28)
        cam_nopool_label1 = cam_nopool[(labels==1)] # (897, 28)
        
        # Average in 1 Fold
        mean_cam_nopool_label0 = np.mean(cam_nopool_label0, axis=0) # (28,)
        mean_cam_nopool_label1 = np.mean(cam_nopool_label1, axis=0) # (28,)
        
        mean_cam_nopools_label0.append(mean_cam_nopool_label0)
        mean_cam_nopools_label1.append(mean_cam_nopool_label1)
        
        cam_nopools_label0.append(cam_nopool_label0)
        cam_nopools_label1.append(cam_nopool_label1)
    
    mean_cam_nopools_label0 = np.array(mean_cam_nopools_label0) # (10, 28)
    mean_cam_nopools_label1 = np.array(mean_cam_nopools_label1) # (10, 28)
    
    cam_nopools_label0 = np.vstack(cam_nopools_label0)
    cam_nopools_label1 = np.vstack(cam_nopools_label1)
    
    # (10, 28), (10, 28),    (9100, 28),   (8970, 28)
    return mean_cam_nopools_label0, mean_cam_nopools_label1, cam_nopools_label0, cam_nopools_label1


def aggregate_grad_cam_combined(output_dir, classify_type):
    f_mean_cam_nopools_label0 = []
    f_mean_cam_nopools_label1 = []    
    f_cam_nopools_label0 = []
    f_cam_nopools_label1 = []
    
    e_mean_cam_nopools_label0 = []
    e_mean_cam_nopools_label1 = []    
    e_cam_nopools_label0 = []
    e_cam_nopools_label1 = []
    
    for i in range(10):
        data = np.load("{}/data/grad_cam_data_combined_ct{}_{}.npz".format(output_dir, classify_type, i),
                       allow_pickle=True)
        labels = data['label'] # (1807,)

        # [fMRI]
        f_cam_nopool = data['f_cam_nopool'] # (1807, 6, 7, 6)
        
        f_cam_nopool_label0 = f_cam_nopool[(labels==0)] # (910, 6, 7, 6))
        f_cam_nopool_label1 = f_cam_nopool[(labels==1)] # (897, 6, 7, 6))    
        
        # Average in 1 Fold
        f_mean_cam_nopool_label0 = np.mean(f_cam_nopool_label0, axis=0) # (6, 7, 6))
        f_mean_cam_nopool_label1 = np.mean(f_cam_nopool_label1, axis=0) # (6, 7, 6))
        
        f_mean_cam_nopools_label0.append(f_mean_cam_nopool_label0)
        f_mean_cam_nopools_label1.append(f_mean_cam_nopool_label1)
        
        f_cam_nopools_label0.append(f_cam_nopool_label0)
        f_cam_nopools_label1.append(f_cam_nopool_label1)

        # [EEG]
        e_cam_nopool = data['e_cam_nopool'] # (1807, 3, 28)
        e_cam_nopool = np.mean(e_cam_nopool, axis=1) # (1807, 28) Take the average of the 3 channels
        
        e_cam_nopool_label0 = e_cam_nopool[(labels==0)] # (910, 28))
        e_cam_nopool_label1 = e_cam_nopool[(labels==1)] # (897, 28))        

        # Average in 1 Fold
        e_mean_cam_nopool_label0 = np.mean(e_cam_nopool_label0, axis=0)
        e_mean_cam_nopool_label1 = np.mean(e_cam_nopool_label1, axis=0)
        
        e_mean_cam_nopools_label0.append(e_mean_cam_nopool_label0)
        e_mean_cam_nopools_label1.append(e_mean_cam_nopool_label1)
        
        e_cam_nopools_label0.append(e_cam_nopool_label0)
        e_cam_nopools_label1.append(e_cam_nopool_label1)
    
    f_mean_cam_nopools_label0 = np.array(f_mean_cam_nopools_label0) # (10, 6, 7, 6))
    f_mean_cam_nopools_label1 = np.array(f_mean_cam_nopools_label1) # (10, 6, 7, 6))
    
    f_cam_nopools_label0 = np.vstack(f_cam_nopools_label0)
    f_cam_nopools_label1 = np.vstack(f_cam_nopools_label1)

    e_mean_cam_nopools_label0 = np.array(e_mean_cam_nopools_label0)
    e_mean_cam_nopools_label1 = np.array(e_mean_cam_nopools_label1)
    
    e_cam_nopools_label0 = np.vstack(e_cam_nopools_label0)
    e_cam_nopools_label1 = np.vstack(e_cam_nopools_label1)            
    
    # (10, 6, 7, 6)), (10, 6, 7, 6)),    (9100, 6, 7, 6)),   (8970, 6, 7, 6)
    # (10, 28)),      (10, 28)),         (9100, 28)),        (8970, 28)
    return f_mean_cam_nopools_label0, f_mean_cam_nopools_label1, f_cam_nopools_label0, f_cam_nopools_label1, \
        e_mean_cam_nopools_label0, e_mean_cam_nopools_label1, e_cam_nopools_label0, e_cam_nopools_label1


def export_grad_cam_eeg_info(file_path,
                             target0, target1,
                             input0, input1,
                             n0, n1,
                             mean0, mean1,
                             std0, std1,
                             se0, se1):

    f = open(file_path, "w")
    f.write("target={}\n".format(target0))
    f.write("input={}\n".format(input0))
    f.write("N={}\n".format(n0))
    f.write("mean={}\n".format(list(mean0)))
    f.write("std={}\n".format(list(std0)))
    f.write("se={}\n".format(list(se0)))
    f.write("\n\n")
    f.write("target={}\n".format(target1))
    f.write("input={}\n".format(input1))
    f.write("N={}\n".format(n1))
    f.write("mean={}\n".format(list(mean1)))
    f.write("std={}\n".format(list(std1)))
    f.write("se={}\n".format(list(se1)))
    f.close()


def export_grad_cam_eeg_sub_sub(output_dir,
                                classify_type, aggregate_type, type,
                                error_type,
                                mean_cams0, mean_cams1,
                                error_cams0, error_cams1):
    if classify_type == FACE_OBJECT:
        target0 = 'face'
        target1 = 'object'
        target0_name = 'Face'
        target1_name = 'Object'
    elif classify_type == MALE_FEMALE:
        target0 = 'male'
        target1 = 'female'
        target0_name = 'Male'
        target1_name = 'Female'
    else:
        target0 = 'artificial'
        target1 = 'natural'
        target0_name = 'Artificial'
        target1_name = 'Natural'

    # x-axis
    xs = (np.arange(0, 28) + 0.5) * (1000/28)
    
    plt.xlabel("time (ms)")
    plt.ylabel("weight")
    plt.xlim([0,1000])
    
    plt.plot(xs, mean_cams0, label=target0_name, color='blue')
    plt.plot(xs, mean_cams1, label=target1_name, color='orange')
    
    d = 5
    plt.errorbar(xs-d, mean_cams0,
                 yerr=error_cams0, capsize=3,
                 markersize=10, ecolor='blue',
                 marker="none", zorder=0, fmt="none", alpha=0.3)
    plt.errorbar(xs+d, mean_cams1,
                 yerr=error_cams1, capsize=
                 3, markersize=10, ecolor='orange',
                 marker="none", zorder=0, fmt="none", alpha=0.3)
    plt.legend()

    if type == "combined":
        additional = "(combined)"
    else:
        additional = ""

    if aggregate_type == "ma":
        aggregation = "(model aggregation)"
    else:
        aggregation = "(trial aggregation)"

    title = "Grad-CAM EEG{} {}/{}\n{}".format(additional, target0_name, target1_name,
                                              aggregation)
    file_path = "{}/{}/eeg_{}_ct{}_mean_{}.pdf".format(output_dir,
                                                       type, aggregate_type, classify_type,
                                                       error_type)
    
    plt.title(title)
    plt.savefig(file_path)
    plt.close()


def export_grad_cam_eeg_sub(output_dir, classify_type, aggregate_type, type, cams0, cams1):
    if classify_type == FACE_OBJECT:
        target0 = 'face'
        target1 = 'object'
        target0_name = 'Face'
        target1_name = 'Object'
    elif classify_type == MALE_FEMALE:
        target0 = 'male'
        target1 = 'female'
        target0_name = 'Male'
        target1_name = 'Female'
    else:
        target0 = 'artificial'
        target1 = 'natural'
        target0_name = 'Artificial'
        target1_name = 'Natural'
        
    mean_cams0 = np.mean(cams0, axis=0)
    mean_cams1 = np.mean(cams1, axis=0)
    
    std_cams0 = np.std(cams0, axis=0, ddof=1)
    std_cams1 = np.std(cams1, axis=0, ddof=1)
    
    se_cams0 = np.std(cams0, axis=0, ddof=1) / np.sqrt(cams0.shape[0])
    se_cams1 = np.std(cams1, axis=0, ddof=1) / np.sqrt(cams1.shape[0])
    
    export_grad_cam_eeg_sub_sub(output_dir,
                                classify_type, aggregate_type, type,
                                "std",
                                mean_cams0, mean_cams1,
                                std_cams0, std_cams1)
    
    export_grad_cam_eeg_sub_sub(output_dir,
                                classify_type, aggregate_type, type,
                                "se",
                                mean_cams0, mean_cams1,
                                se_cams0, se_cams1)
    
    file_path = "{}/{}/eeg_{}_ct{}.txt".format(output_dir,
                                               type, aggregate_type, classify_type)
    export_grad_cam_eeg_info(file_path,
                             target0, target1,
                             target0, target1,
                             cams0.shape[0], cams1.shape[0],
                             mean_cams0, mean_cams1,
                             std_cams0, std_cams1,
                             se_cams0, se_cams1)


def export_grad_cam_fmri_sub(output_dir, classify_type, aggregate_type, type, cams0, cams1):
    if classify_type == FACE_OBJECT:
        target0 = 'face'
        target1 = 'object'
        target0_name = 'Face'
        target1_name = 'Object'
    elif classify_type == MALE_FEMALE:
        target0 = 'male'
        target1 = 'female'
        target0_name = 'Male'
        target1_name = 'Female'
    else:
        target0 = 'artificial'
        target1 = 'natural'
        target0_name = 'Artificial'
        target1_name = 'Natural'
    
    mean_cams0 = np.mean(cams0, axis=0)
    mean_cams1 = np.mean(cams1, axis=0)
    std_cams0 = np.std(cams0, axis=0, ddof=1)
    std_cams1 = np.std(cams1, axis=0, ddof=1)
    se_cams0 = np.std(cams0, axis=0, ddof=1) / np.sqrt(cams0.shape[0])
    se_cams1 = np.std(cams1, axis=0, ddof=1) / np.sqrt(cams1.shape[0])  
        
    if type == "combined":
        additional = "(combined)"
    else:
        additional = ""

    if aggregate_type == "ma":
        aggregation = "(model aggregation)"
    else:
        aggregation = "(trial aggregation)"
        
    export_fcam(mean_cams0, 
                "Grad-CAM MRI{} {}\n{}".format(additional, target0_name, aggregation),
                "{}/{}/fmri_{}_ct{}_{}_mean.pdf".format(output_dir,
                                                        type, aggregate_type, classify_type, target0))
    export_fcam(mean_cams1, 
                "Grad-CAM MRI{} {}\n{}".format(additional, target1_name, aggregation),
                "{}/{}/fmri_{}_ct{}_{}_mean.pdf".format(output_dir,
                                                        type, aggregate_type, classify_type, target1))
    
    export_fcam(std_cams0, 
                "Grad-CAM MRI STD{} {}\n{}".format(additional, target0_name, aggregation),
                "{}/{}/fmri_{}_ct{}_{}_std.pdf".format(output_dir,
                                                       type, aggregate_type, classify_type, target0))
    export_fcam(std_cams1, 
                "Grad-CAM MRI STD{} {}\n{}".format(additional, target1_name, aggregation),
                "{}/{}/fmri_{}_ct{}_{}_std.pdf".format(output_dir,
                                                       type, aggregate_type, classify_type, target1))
    
    export_fcam(se_cams0, 
                "Grad-CAM MRI SE{} {}\n{}".format(additional, target0_name, aggregation),
                "{}/{}/fmri_{}_ct{}_{}_se.pdf".format(output_dir,
                                                      type, aggregate_type, classify_type, target0))
    export_fcam(se_cams1, 
                "Grad-CAM MRI SE{} {}\n{}".format(additional, target1_name, aggregation),
                "{}/{}/fmri_{}_ct{}_{}_se.pdf".format(output_dir,
                                                      type, aggregate_type, classify_type, target1))


def export_grad_cam_eeg(output_dir, classify_type):
    out = aggregate_grad_cam(output_dir, classify_type, type="eeg")
    mean_cams0, mean_cams1, cams0, cams1 = out

    # Model aggregation
    export_grad_cam_eeg_sub(output_dir, classify_type, "ma", "eeg", mean_cams0, mean_cams1)

    # Trial aggregation
    export_grad_cam_eeg_sub(output_dir, classify_type, "ta", "eeg", cams0, cams1)


def export_grad_cam_fmri(output_dir, classify_type):
    out = aggregate_grad_cam(output_dir, classify_type, type='fmri')
    mean_cams0, mean_cams1, cams0, cams1 = out

    # Model aggregation
    export_grad_cam_fmri_sub(output_dir, classify_type, "ma", "fmri", mean_cams0, mean_cams1)
    
    # Trial aggregation
    export_grad_cam_fmri_sub(output_dir, classify_type, "ta", "fmri", cams0, cams1)


def export_grad_cam_combined(output_dir, classify_type):
    out = aggregate_grad_cam_combined(output_dir, classify_type)
    f_mean_cams0, f_mean_cams1, f_cams0, f_cams1, \
        e_mean_cams0, e_mean_cams1, e_cams0, e_cams1 = out

    # Model aggregation
    export_grad_cam_fmri_sub(output_dir, classify_type, "ma", "combined", f_mean_cams0, f_mean_cams1)
    export_grad_cam_eeg_sub(output_dir, classify_type, "ma", "combined", e_mean_cams0, e_mean_cams1)
    
    # Trial aggregation
    export_grad_cam_fmri_sub(output_dir, classify_type, "ta", "combined", f_cams0, f_cams1)
    export_grad_cam_eeg_sub(output_dir, classify_type, "ta", "combined", e_cams0, e_cams1)


def export_eeg(output_dir):
    export_grad_cam_eeg(output_dir, FACE_OBJECT)
    export_grad_cam_eeg(output_dir, MALE_FEMALE)
    export_grad_cam_eeg(output_dir, ARTIFICIAL_NATURAL)

    
def export_fmri(output_dir):
    export_grad_cam_fmri(output_dir, FACE_OBJECT)
    export_grad_cam_fmri(output_dir, MALE_FEMALE)
    export_grad_cam_fmri(output_dir, ARTIFICIAL_NATURAL)


def export_combined(output_dir):
    export_grad_cam_combined(output_dir, FACE_OBJECT)
    export_grad_cam_combined(output_dir, MALE_FEMALE)
    export_grad_cam_combined(output_dir, ARTIFICIAL_NATURAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="./grad_cam_results")
    args = parser.parse_args()

    output_dir = args.output_dir

    if not os.path.exists(output_dir + "/eeg"):
        os.makedirs(output_dir + "/eeg")
    if not os.path.exists(output_dir + "/fmri"):
        os.makedirs(output_dir + "/fmri")
    if not os.path.exists(output_dir + "/combined"):
        os.makedirs(output_dir + "/combined")
    
    export_eeg(output_dir)
    export_fmri(output_dir)
    export_combined(output_dir)
