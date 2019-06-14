# Use: python train.py flowers --arch vgg11 --save_dir ../model --learning_rate 0.001 --hidden_units 256 --epochs 4 --gpu

import sys
import argparse

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch import nn
import torchvision.models as models
from torchvision import datasets, transforms

import datetime

def get_args():
    args = {}
   
    parser = argparse.ArgumentParser(description='Image classifier')
    parser.add_argument("image_dir",
                        help = 'directory of images (train and test)')
    parser.add_argument('--arch', type=str, choices=["vgg11", "vgg13", "vgg16"], required=True,
                       help = 'arch to choose (vgg11 or vgg13 or vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help = 'Coeficient of learning rate')
    parser.add_argument('--hidden_units', type=int, default=256,
                       help = 'Number of hidden units')
    parser.add_argument('--epochs', type=int, default=4,
                       help = 'Number of epochs')
    parser.add_argument('--gpu', action='store_true',
                       help = 'Use this flag if you want to use gpu')
    parser.add_argument('--save_dir', type=str, default="../model",
                       help = 'Directory destination of pth file')
    parsed_args = parser.parse_args()
    
    args = vars(parsed_args)

    return args

def get_dataloaders(train_dir, test_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    dataloaders = {}
    
    trainset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    
    testset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)
    
    dataloaders["train"] = {
        "dataset": trainset,
        "loader": trainloader
    }    
    
    dataloaders["test"] = {
        "dataset": testset,
        "loader": testloader
    }    
    
    return dataloaders

def create_model(arch):
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    return model


def train_model(model, trainloader, testloader, learning_rate, hidden_units, epochs, gpu):
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device);
    
    epochs = epochs
    steps = 4
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
                
    return model

def save_model(model, trainset, save_dir, arch, hidden_units):
    name_file = "classsifier_" + str(datetime.datetime.now()).replace(" ","_") + ".pth"

    model.class_to_idx = trainset.class_to_idx
    torch.save({'arch': arch,
                'hidden_units': hidden_units
                'state_dict': model.state_dict(), 
                'class_to_idx': model.class_to_idx}, 
                save_dir + '/' + name_file)
    
def main():
    args = get_args()
    
    train_dir = args["image_dir"] + "/train"
    test_dir = args["image_dir"] + "/test"
    dataloaders = get_dataloaders(train_dir, test_dir)
    model = create_model(args["arch"])
    model = train_model(
        model, 
        dataloaders["train"]["loader"], 
        dataloaders["test"]["loader"],
        args["learning_rate"],
        args["hidden_units"],
        args["epochs"],
        args["gpu"])
    
    save_model(model, dataloaders["train"]["dataset"], args["save_dir"], args["arch"], args["hidden_units"])
    
if __name__ == "__main__":
    main()