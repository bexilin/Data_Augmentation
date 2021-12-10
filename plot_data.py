import argparse
import json
from itertools import combinations
import matplotlib.pyplot as plt
import os
import numpy as np

def Load_Dir(params,exam_classes,image_type,classify_struct):
    for i in range(len(exam_classes)-1):
        if i == 0:
            dir_name = exam_classes[i] + "_" + exam_classes[i+1]
        else:
            dir_name = dir_name + "_" + exam_classes[i+1]
    classes_dir = str(len(exam_classes))
    struct_dir = image_type + "_" + classify_struct
    load_dir = os.path.join(params["data_dir"],classes_dir,dir_name,struct_dir)
    return load_dir

parser = argparse.ArgumentParser(description="Run experiments")
parser.add_argument(
    "-C",
    "--config-file",
    dest="config_file",
    type=open,
    required=True,
    help="configuration file for experiment parameters",
)

args = parser.parse_args()
params = json.load(args.config_file)

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
save_idx_notation = True
struct_idx_notation = []
count = 0
two_class_results = []
five_class_results = []
for image_type in params["image_type"]:
    for classify_struct in params["classify_struct"]:
        struct_idx_notation.append([str(count),image_type + "_" + classify_struct])

        mean_accs = []
        if save_idx_notation:
            idx_notation = []
        
        # plot results of 2-class classification
        for idx, exam_classes in enumerate(combinations(params["classes"], 2)):
            exam_classes = list(exam_classes)
            load_dir = Load_Dir(params,exam_classes,image_type,classify_struct)
            data = np.load(os.path.join(load_dir,"max_val_accuracies.npy"))
            mean_acc = np.mean(data)
            mean_accs.append(mean_acc)
            if save_idx_notation:
                notation = [str(idx)]
                notation.append(params["class_note"][exam_classes[0]])
                notation.append(params["class_note"][exam_classes[1]])
                idx_notation.append(notation)

        two_class_results.append([str(count),str(np.mean(mean_accs))])
        ax.plot([x for x in range(len(mean_accs))],mean_accs,label=count)
        
        if save_idx_notation:
            np.savetxt(os.path.join(params["data_dir"],"2_classify_class_notes.csv"),
                       idx_notation,fmt="%s",delimiter=",")

        # save results of 5-class classification
        load_dir_2 = Load_Dir(params,params["classes"],image_type,classify_struct)
        data_2 = np.load(os.path.join(load_dir_2,"max_val_accuracies.npy"))
        five_class_results.append([str(count),str(np.mean(data_2))])

        save_idx_notation = False
        count = count + 1

np.savetxt(os.path.join(params["data_dir"],"2_classify_struct_notes.csv"),
           struct_idx_notation,fmt="%s",delimiter=",")

np.savetxt(os.path.join(params["data_dir"],"2_classify_results.csv"),
           two_class_results,fmt="%s",delimiter=",")

np.savetxt(os.path.join(params["data_dir"],"5_classify_results.csv"),
           five_class_results,fmt="%s",delimiter=",")

ax.legend()
ax.xaxis.set_ticks(np.arange(0, 10, 1))
ax.set_xlabel("Classification")
ax.set_ylabel("Prediction accuracy")
ax.set_title("2-class classification results")

fig.savefig(os.path.join(params["data_dir"],"2_class_results.png"))

