import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import argparse

ap = argparse.ArgumentParser(description='fun_utility.py')
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
model_name = pa.arch
dropout = pa.dropout
epochs = pa.epochs

### load all the data set
def load_data(where  = "./flowers"):
    data_dir = where
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_datasets  = datasets.ImageFolder(test_dir,  transform = test_transforms) 
    image_datasets = [train_datasets, valid_datasets, test_datasets]

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True)

    return trainloader, validloader, testloader, train_datasets.class_to_idx


## set up the model
def model_setup(model_name='vgg11',dropout=dropout, hidden_layer1 = 120, lr = lr, power='gpu'):
    arch = {"vgg11":25088,
        "densenet121":1024}
    
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_in_features = model.fc.in_features
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
        num_in_features = model.classifier[0].in_features
    else:
        print("Unknown model...")
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(arch[structure], hidden_layer1),
                          nn.ReLU(),
                          nn.Dropout(dropout),
                          nn.Linear(hidden_layer1, 102),
                          nn.LogSoftmax(dim =1))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )
    
    device = torch.device('cuda' if torch.cuda.is_available() and power == 'gpu' else 'cpu')
    model.to(device)
    
    return model, optimizer, criterion
   
    
#model, optimizer, criterion = model_setup()

def train_network(model, criterion, optimizer, epochs, trainloader, validloader,  print_every=20, power='gpu'):
    steps = 0 
    running_loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() and power == 'gpu' else 'cpu')
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps +=1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
            if steps%print_every ==0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    return model, optimizer
                
def test_accuracy(model, criterion, testloader, power='gpu'):
    model.eval()
    test_loss = 0
    accuracy = 0
    device = torch.device('cuda' if torch.cuda.is_available() and power == 'gpu' else 'cpu')
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print(f"Test accuracy: {accuracy/len(testloader):.3f}")
                
def save_checkpoint(model, optimizer, structure, epochs, ctx, path ='checkpoint.pth'):
    model.class_to_idx = ctx
    model.cpu
    checkpoint = {'state_dict': model.state_dict(), 
              'mapping_class': model.class_to_idx, 
             'optimizer_state': optimizer.state_dict, 
              'epochs': epochs, 
            'structure': structure}
    torch.save(checkpoint, path)
    
def load_checkpoint(dropout, lr, filepath='checkpoint.pth'):
    checkpoint = torch.load(filepath)
    structure = checkpoint['structure']
    if structure != 'vgg11':
        print('Model strucutre is not what it is trained on, VGG11')
        return None
    else:
        model = models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
               
    
        classifier = nn.Sequential(nn.Linear(25088, 4096),
                          nn.ReLU(),
                          nn.Dropout(dropout),
                          nn.Linear(4096, 102),
                          nn.LogSoftmax(dim =1))

        model.classifier = classifier
        criterion = nn.NLLLoss()
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['mapping_class']
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        optimizer.state_dict = checkpoint['optimizer_state']
        criterion = nn.NLLLoss()
        return model, optimizer, criterion
    