from network_model import Aug_Model
from data_loader import Data_Handler
import argparse
import json
import torch
from torch import nn
from torch import optim

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
print("epoches: ",params["epoches"])
print("alpha: ",params["alpha"])
print("beta: ",params["beta"],"\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = Data_Handler(params["data_dir"],params["classes"])
model = Aug_Model(len(params["classes"])).to(device)
optimizer = optim.Adam(model.parameters(),lr=params["lr"])
CEL = nn.CrossEntropyLoss()
MSE = nn.MSELoss()

val_inputs, val_labels = data.GetValSet()

val_X = torch.from_numpy(val_inputs).to(torch.float).to(device=device)
val_Y = torch.from_numpy(val_labels).to(torch.float).to(device=device)

def Accuracy(pred, truth):
    pred_labels = torch.argmax(pred,dim=1)
    truth_labels = torch.argmax(truth,dim=1)
    diff = pred_labels - truth_labels
    correctness = 0.0
    for idx in range(len(diff)):
        if diff[idx] == 0:
            correctness = correctness + 1
    return correctness / len(diff)

max_train_acc = 0.0
max_eval_acc = 0.0
for epoch in range(params["epoches"]):
    print("epoch: ",epoch)

    optimizer.zero_grad()
    input_samples, target_samples, labels = data.Samples()
    
    X = torch.from_numpy(input_samples).to(torch.float).to(device=device)
    T = torch.from_numpy(target_samples).to(torch.float).to(device=device)
    Y = torch.from_numpy(labels).to(torch.float).to(device=device)
    
    pred = model(X)
    L_c = CEL(pred,Y)
    L_a = MSE(model.getTargetImg(),T)
    Loss = params["alpha"]*L_c + params["beta"]*L_a
    
    print("L_c: ", L_c)
    print("L_a: ", L_a)
    print("Loss: ", Loss, "\n")

    train_acc = Accuracy(pred,Y)
    print("Training accuracy: ",train_acc)
    if train_acc > max_train_acc:
        max_train_acc = train_acc

    pred = model(val_X)
    
    eval_acc = Accuracy(pred,val_Y)
    print("Evaluation accuracy: ", eval_acc,"\n")
    if eval_acc > max_eval_acc:
        max_eval_acc = eval_acc

    Loss.backward()
    optimizer.step()

print("max training accuracy: ", max_train_acc)
print("max evaluation accuracy: ", max_eval_acc,"\n")

# Evaluation
#val_inputs, val_labels = data.GetValSet()

#val_X = torch.from_numpy(val_inputs).to(torch.float).to(device=device)
#val_Y = torch.from_numpy(val_labels).to(torch.float).to(device=device)

#pred = model(val_X)

#print("Evaluation accuracy: ", Accuracy(pred,val_Y),"\n")
