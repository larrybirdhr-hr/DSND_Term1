# Imports here

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image

import argparse

import fun_utility

ap = argparse.ArgumentParser(description='Train.py')

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.3)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
ap.add_argument('--arch', dest="arch", action="store", default="vgg11", type = str)
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)



pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
model_name = pa.arch
dropout = pa.dropout
epochs = pa.epochs
hidden_layer1 = pa.hidden_units
power = pa.gpu

    



trainloader, validloader, testloader, ctx = fun_utility.load_data(where)

model, optimizer, criterion = fun_utility.model_setup(model_name, dropout, hidden_layer1, lr, power)
model, optimizer = fun_utility.train_network(model, criterion, optimizer, epochs, trainloader, validloader, 20, power)


fun_utility.save_checkpoint(model, optimizer, model_name, epochs, ctx, path)


#model, optimizer, criterion = fun_utility.load_checkpoint(dropout, lr, path)
#model.to('cuda')
fun_utility.test_accuracy(model, criterion, testloader, power)


