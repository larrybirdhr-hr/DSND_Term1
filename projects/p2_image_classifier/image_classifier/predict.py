import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import json
from PIL import Image
import argparse

import fun_utility

ap = argparse.ArgumentParser(description='predict.py')
ap.add_argument('input_img', default='flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.3)

pa = ap.parse_args()
input_img = pa.input_img
number_of_outputs = pa.top_k
lr = pa.learning_rate

dropout = pa.dropout
path = pa.checkpoint
power = pa.gpu
cat_name = pa.category_names

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model

    im = Image.open(image)
   
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    im= transform(im)
    
    return im

def predict(input_img, model, topk=number_of_outputs, power = 'gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() and power == 'gpu' else 'cpu')

    model.to(device)
    model.eval();
    img = process_image(input_img)
    ## this is an error and checked it on the discussion board to fix it
    img = img.unsqueeze_(0)
    img = img.float()
    
    if power =='gpu':
        with torch.no_grad():
            log_prob = model.forward(img.cuda())
    else:
        with torch.no_grad():
            log_prob = model.forward(img)
        
    prob = torch.exp(log_prob)
    
    return prob.topk(topk)

trainloader, validloader, testloader, ctx  = fun_utility.load_data()

model, optimizer, criterion = fun_utility.load_checkpoint(dropout, lr, path)


with open(cat_name, 'r') as f:
    cat_to_name = json.load(f)   
    
prob = predict(input_img, model, number_of_outputs, power)
a = np.array(prob[0][0])
b = [cat_to_name[str(index+1)] for index in np.array(prob[1][0])]

i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(b[i], a[i]))
    i += 1