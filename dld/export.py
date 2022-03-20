import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import os

def export_bar_graph(accuracies, errors, title, save_dir):
    x = np.array(['F/O\nTrain',
                  'F/O\nValid',
                  'M/F\nTrain',
                  'M/F\nValid',
                  'A/N\nTrain',
                  'A/N\nValid'])
    bar_x_position = np.array([0,1,2+1,3+1,4+2,5+2])
    
    plt.bar(bar_x_position, accuracies, yerr=errors, width=0.6)
    plt.xticks(bar_x_position, x)

    plt.title(title)
    plt.ylabel('Decode accuracy (%)')
    plt.ylim([0.0, 100.0])
    
    # Additional lines
    plt.hlines([50.0], -1, 8, "blue", linestyles='dashed')
    
    plt.xlim([-1, 8])

    file_path = save_dir + "/result.png"
    plt.savefig(file_path, format="png", dpi=288)
    plt.close()


def load_resuls(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    train_mean      = float(lines[0].strip().split("=")[1])
    validation_mean = float(lines[1].strip().split("=")[1])
    train_std       = float(lines[2].strip().split("=")[1])
    validation_std  = float(lines[3].strip().split("=")[1])
    return (train_mean, validation_mean, train_std, validation_std)


def export_results(save_dir, title):
    all_results = []

    accuracies = []
    errors = []
    
    for classify_type in range(3):
        path = save_dir + "/result_ct{}.txt".format(classify_type)
        results = load_resuls(path)
        accuracies.append(results[0])
        accuracies.append(results[1])
        errors.append(results[2])
        errors.append(results[3])

    export_bar_graph(accuracies, errors, title, save_dir)

    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="saved")
    args = parser.parse_args()
    save_dir = args.save_dir
    title = "exp: {}".format(save_dir)
    export_results(save_dir, title)
