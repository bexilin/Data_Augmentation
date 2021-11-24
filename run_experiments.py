from network_model import Aug_Model
from data_loader import Data_Handler
import argparse
import json
import torch
from torch import nn
from torch import optim
import numpy as np
import os

def Make_Save_Dir(params):
    for i in range(len(params["classes"])-1):
        if i == 0:
            dir_name = params["classes"][i] + "_" + params["classes"][i+1]
        else:
            dir_name = dir_name + "_" + params["classes"][i+1]
    save_dir = params["data_dir"] + "/" + str(len(params["classes"])) + "/" + dir_name
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

print("\n====== Experiment setup ======\n")
print("classes: ",params["classes"])
print("learning rate: ",params["lr"])
print("epochs: ",params["epochs"])
print("alpha: ",params["alpha"])
print("beta: ",params["beta"],"\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_dir = params["data_dir"] + "/tiny-imagenet-200"
save_dir = Make_Save_Dir(params)
data = Data_Handler(dataset_dir, params["classes"])
model = Aug_Model(len(params["classes"])).to(device)
optimizer = optim.Adam(model.parameters(),lr=params["lr"])
CEL = nn.CrossEntropyLoss()
MSE = nn.MSELoss()

val_inputs, val_labels = data.GetValSet()

val_X = torch.from_numpy(val_inputs).to(torch.float).to(device=device)
val_Y = torch.from_numpy(val_labels).to(torch.float).to(device=device)

max_eval_accs = []
max_max_eval_accs = 0.0
for repeat in range(params["repeats"]):
    print("=== training counts ===: ",repeat,"\n")
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

        pred = model(val_X)
        
        eval_acc = Accuracy(pred,val_Y)
        # print("Evaluation accuracy: ", eval_acc,"\n")
        if eval_acc > max_eval_acc:
            max_eval_acc = eval_acc

        if eval_acc > max_max_eval_accs:
            max_max_eval_accs = eval_acc
            torch.save(model.state_dict(), save_dir+"/best_model_params.pt")

        Loss.backward()
        optimizer.step()

    print("max training accuracy: ", max_train_acc)
    print("max validation accuracy: ", max_eval_acc,"\n")
    max_eval_accs.append(max_eval_acc)

mean_max_eval_accs = np.mean(max_eval_accs)
SE_max_eval_accs = np.std(max_eval_accs) / np.sqrt(len(max_eval_accs))
np.save(save_dir+"/max_val_accuracies.npy",max_eval_accs)
np.savetxt(save_dir+"/mean_and_SE.txt",[mean_max_eval_accs,SE_max_eval_accs],delimiter=",")



# Evaluation
#val_inputs, val_labels = data.GetValSet()

#val_X = torch.from_numpy(val_inputs).to(torch.float).to(device=device)
#val_Y = torch.from_numpy(val_labels).to(torch.float).to(device=device)

#pred = model(val_X)

#print("Evaluation accuracy: ", Accuracy(pred,val_Y),"\n")
