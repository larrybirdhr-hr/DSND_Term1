# Imports here

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
#from matplotlib.ticker import FormatStrFormatter
import argparse

import fun_utility

ap = argparse.ArgumentParser(description='Train.py')

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.3)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=12)
ap.add_argument('--arch', dest="arch", action="store", default="vgg11", type = str)


pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
epochs = pa.epochs


    



trainloader, validloader, testloader, ctx = fun_utility.load_data(where)

# model, optimizer, criterion = fun_utility.model_setup()
# model, optimizer = fun_utility.train_network(model, criterion, optimizer, epochs, trainloader=trainloader, validloader= validloader)

#



model, optimizer, criterion = fun_utility.load_checkpoint(dropout, lr)
model.to('cuda')
fun_utility.test_accuracy(model, criterion, testloader)


