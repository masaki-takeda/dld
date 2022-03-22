import numpy as np
import matplotlib.pyplot as plt

from dataset import FACE_OBJECT, MALE_FEMALE, ARTIFICIAL_NATURAL


def normalize(cam):
    cam = cam - np.min(cam)
    max_value = np.max(cam)
    if max_value == 0.0:
        return np.zeros_like(cam)
    else:
        cam = cam / max_value
        return cam


def export_fcam(fcam, title, file_path):
    fig, axes = plt.subplots(6, 1, figsize=(4.8, 6.4*4), tight_layout=True)

    for i in range(6):
        ax = axes[i]
        if i == 0:
            ax.set_title(title)
        im = ax.imshow(fcam[i],
                       interpolation='bicubic',
                       vmin=0,
                       vmax=1, 
               cmap='jet')
    
    axpos = axes[0].get_position()
    cbar_ax = fig.add_axes([0.87, axpos.y0 + 0.105 , 0.02, axpos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(file_path)
    plt.close()


def export_ecam(ecam, title, file_path):
    # x-axis
    xs = (np.arange(0, 28) + 0.5) * (1000/28)
    
    plt.title(title)
    plt.xlabel("time (ms)")
    plt.ylabel("weight")
    plt.xlim([0,1000])
    plt.plot(xs, ecam)
    plt.savefig(file_path)
    plt.close()


def export_grad_cam_fmri(classify_type):
    data = np.load("./grad_cam_results/grad_cam_data_fmri_ct{}_0.npz".format(classify_type))
    #cam = data['cam'] # (1807, 6, 7, 6)
    cam_nopool = data['cam_nopool'] # (1807, 6, 7, 6)
    labels = data['label'] # (*,)
    #predicted_labels = data['predicted_label']
    predicted_probs = data['predicted_prob']

    cam_nopool_label0 = normalize(np.mean(cam_nopool[(labels==0)], axis=0))
    cam_nopool_label1 = normalize(np.mean(cam_nopool[(labels==1)], axis=0))

    """
    cam_nopool_label0_high = normalize(np.mean(cam_nopool[(labels==0) & (predicted_probs < 0.25)],
                                               axis=0))
    cam_nopool_label1_high = normalize(np.mean(cam_nopool[(labels==1) & (predicted_probs < 0.75)],
                                               axis=0))
    """

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
    
    export_fcam(cam_nopool_label0, 
                "Grad-CAM MRI {}".format(target0_name),
                "grad_cam_results/grad_cam_fmri_{}.pdf".format(target0))
    export_fcam(cam_nopool_label1, 
                "Grad-CAM MRI {}".format(target1_name),
                "grad_cam_results/grad_cam_fmri_{}.pdf".format(target1))
    
    """
    export_fcam(cam_nopool_label0, 
                "Grad-CAM MRI {} (>75%)".format(target0_name),
                "grad_cam_results/grad_cam_fmri_{}_high.pdf".format(target0))
    export_fcam(cam_nopool_label1, 
                "Grad-CAM MRI {} (>75%)".format(target1_name),
                "grad_cam_results/grad_cam_fmri_{}_high.pdf".format(target1))
    """


def export_grad_cam_eeg(classify_type):
    data = np.load("./grad_cam_results/grad_cam_data_eeg_ct{}_0.npz".format(classify_type))
    #cam = data['cam'] # (1807, 3, 28)
    cam_nopool = data['cam_nopool'] # (1807, 3, 28)
    cam_nopool = np.mean(cam_nopool, axis=1) # (1807, 28) Take the average of the 3 channels
    
    labels = data['label'] # (1807,)
    #predicted_labels = data['predicted_label']
    predicted_probs = data['predicted_prob']
    
    cam_nopool_label0 = normalize(np.mean(cam_nopool[(labels==0)], axis=0))
    cam_nopool_label1 = normalize(np.mean(cam_nopool[(labels==1)], axis=0))

    """
    cam_nopool_label0_high = normalize(np.mean(cam_nopool[(labels==0) & (predicted_probs < 0.25)],
                                               axis=0))
    cam_nopool_label1_high = normalize(np.mean(cam_nopool[(labels==1) & (predicted_probs < 0.75)],
                                               axis=0))
    """

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
    
    export_ecam(cam_nopool_label0, 
                "Grad-CAM MRI {}".format(target0_name),
                "grad_cam_results/grad_cam_eeg_{}.pdf".format(target0))
    export_ecam(cam_nopool_label1, 
                "Grad-CAM MRI {}".format(target1_name),
                "grad_cam_results/grad_cam_eeg_{}.pdf".format(target1))


def export_grad_cam_combined(classify_type):
    data = np.load("./grad_cam_results/grad_cam_data_combined_ct{}_0.npz".format(classify_type))
    f_cam_nopool = data['f_cam_nopool']
    e_cam_nopool = data['e_cam_nopool'] # (1807, 3, 28)
    e_cam_nopool = np.mean(e_cam_nopool, axis=1) # (1807, 28) Take the average of the 3 channels
    
    labels = data['label'] # (*,)

    f_cam_nopool_label0 = normalize(np.mean(f_cam_nopool[(labels==0)], axis=0))
    f_cam_nopool_label1 = normalize(np.mean(f_cam_nopool[(labels==1)], axis=0))

    e_cam_nopool_label0 = normalize(np.mean(e_cam_nopool[(labels==0)], axis=0))
    e_cam_nopool_label1 = normalize(np.mean(e_cam_nopool[(labels==1)], axis=0))    

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
    
    export_fcam(f_cam_nopool_label0, 
                "Grad-CAM Combined MRI {}".format(target0_name),
                "grad_cam_results/grad_cam_combined_fmri_{}.pdf".format(target0))
    export_fcam(f_cam_nopool_label1, 
                "Grad-CAM Combined MRI {}".format(target1_name),
                "grad_cam_results/grad_cam_combined_fmri_{}.pdf".format(target1))

    export_ecam(e_cam_nopool_label0, 
                "Grad-CAM Combined EEG {}".format(target0_name),
                "grad_cam_results/grad_cam_combined_eeg_{}.pdf".format(target0))
    export_ecam(e_cam_nopool_label1, 
                "Grad-CAM Combined EEG {}".format(target1_name),
                "grad_cam_results/grad_cam_combined_eeg_{}.pdf".format(target1))

    
def export_fmri():
    export_grad_cam_fmri(FACE_OBJECT)
    export_grad_cam_fmri(MALE_FEMALE)
    export_grad_cam_fmri(ARTIFICIAL_NATURAL)

    
def export_eeg():
    export_grad_cam_eeg(FACE_OBJECT)
    export_grad_cam_eeg(MALE_FEMALE)
    export_grad_cam_eeg(ARTIFICIAL_NATURAL)


def export_combined():
    export_grad_cam_combined(FACE_OBJECT)
    export_grad_cam_combined(MALE_FEMALE)
    export_grad_cam_combined(ARTIFICIAL_NATURAL)


if __name__ == '__main__':
    export_eeg()
    export_fmri()
    export_combined()
