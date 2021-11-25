from network_model import Aug_Model
from data_loader import Data_Handler
import argparse
import json
import torch
from torch import nn
from torch import optim
import numpy as np
import os
from itertools import combinations

def Make_Save_Dir(params,exam_classes,image_type,classify_struct):
    for i in range(len(exam_classes)-1):
        if i == 0:
            dir_name = exam_classes[i] + "_" + exam_classes[i+1]
        else:
            dir_name = dir_name + "_" + exam_classes[i+1]
    classes_dir = str(len(exam_classes))
    struct_dir = image_type + "_" + classify_struct
    save_dir = os.path.join(params["data_dir"],classes_dir,dir_name,struct_dir)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def Accuracy(pred, truth):
    pred_labels = torch.argmax(pred,dim=1)
    truth_labels = torch.argmax(truth,dim=1)
    diff = pred_labels - truth_labels
    correctness = 0.0
    for idx in range(len(diff)):
        if diff[idx] == 0:
            correctness = correctness + 1
    return correctness / len(diff)

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

for exam_classes in combinations(params["classes"], params["n-classfy"]):
    exam_classes = list(exam_classes)
    for image_type in params["image_type"]:
        for classify_struct in params["classify_struct"]:

            grayscale = True if image_type == "grayscale" else False
            
            # If we use the Aug_and_classify structure in validation,
            # then the validation images are doubled and stacked, and
            # use the same NN structure as that used in training
            stack = True if classify_struct == "Aug_and_classify" else False
            same_as_train = True if classify_struct == "Aug_and_classify" else False

            print("\n====== Experiment setup ======\n")
            print("classes: ",exam_classes)
            print("image type: ",image_type)
            print("NN classify structure: ",classify_struct)
            print("learning rate: ",params["lr"])
            print("epochs: ",params["epochs"])
            print("alpha: ",params["alpha"])
            print("beta: ",params["beta"],"\n")

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # torch.manual_seed(0)
            # np.random.seed(0)

            dataset_dir = params["data_dir"] + "/tiny-imagenet-200"
            save_dir = Make_Save_Dir(params,exam_classes,image_type,classify_struct)
            data = Data_Handler(dataset_dir, exam_classes, grayscale=grayscale)
            CEL = nn.CrossEntropyLoss()
            MSE = nn.MSELoss()

            val_inputs, val_labels = data.GetValSet(stack=stack)

            val_X = torch.from_numpy(val_inputs).to(torch.float).to(device=device)
            val_Y = torch.from_numpy(val_labels).to(torch.float).to(device=device)

            max_eval_accs = []
            max_max_eval_accs = 0.0
            for repeat in range(params["repeats"]):
                model = Aug_Model(len(exam_classes), grayscale=grayscale).to(device)    
                optimizer = optim.Adam(model.parameters(),lr=params["lr"])
                print("=== training model counts ===: ",repeat,"\n")
                max_train_acc = 0.0
                max_eval_acc = 0.0
                for epoch in range(params["epochs"]):
                    if epoch % 10 == 0:
                        print("running epoch: ",epoch)

                    optimizer.zero_grad()
                    input_samples, target_samples, labels = data.Samples()
                    
                    X = torch.from_numpy(input_samples).to(torch.float).to(device=device)
                    T = torch.from_numpy(target_samples).to(torch.float).to(device=device)
                    Y = torch.from_numpy(labels).to(torch.float).to(device=device)
                    
                    pred = model(X)
                    L_c = CEL(pred,Y)
                    L_a = MSE(model.getTargetImg(),T)
                    Loss = params["alpha"]*L_c + params["beta"]*L_a
                    
                    # print("L_c: ", L_c)
                    # print("L_a: ", L_a)
                    # print("Loss: ", Loss, "\n")

                    train_acc = Accuracy(pred,Y)
                    # print("Training accuracy: ",train_acc)
                    if train_acc > max_train_acc:
                        max_train_acc = train_acc

                    pred_2 = model(val_X,same_as_train)
                    
                    eval_acc = Accuracy(pred_2,val_Y)
                    # print("Evaluation accuracy: ", eval_acc,"\n")
                    if eval_acc > max_eval_acc:
                        max_eval_acc = eval_acc

                    #if eval_acc > max_max_eval_accs:
                    #    max_max_eval_accs = eval_acc
                    #    torch.save(model.state_dict(), save_dir+"/best_model_params.pt")

                    Loss.backward()
                    optimizer.step()

                print("\nmax training accuracy: ", max_train_acc)
                print("max validation accuracy: ", max_eval_acc,"\n")
                max_eval_accs.append(max_eval_acc)

            mean_max_eval_accs = np.mean(max_eval_accs)
            SE_max_eval_accs = np.std(max_eval_accs) / np.sqrt(len(max_eval_accs))
            print("=== Validation accuracies ===")
            print("Mean: ",mean_max_eval_accs," SE: ",SE_max_eval_accs,"\n")
            np.save(save_dir+"/max_val_accuracies.npy",max_eval_accs)
            np.savetxt(save_dir+"/mean_and_SE.txt",[mean_max_eval_accs,SE_max_eval_accs],delimiter=",")
