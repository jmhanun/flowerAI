# python predict.py ./flowers/test/2/image_05100.jpg ../model/classsifier_2019-05-16_03:16:06.790767.pth --topk 5 --category_names ./cat_to_name.json --gpu

import argparse

import json
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from PIL import Image

def get_args():
    args = {}
    
    parser = argparse.ArgumentParser(description='Image prediction')
    parser.add_argument("input",
                       help = 'file to predict')
    parser.add_argument("checkpoint",
                       help = 'model backup')
    parser.add_argument('--topk', type=int, default=5,
                       help = 'Integer - how many results')
    parser.add_argument('--category_names', type=str, default=None,
                       help = 'names of category')
    parser.add_argument('--gpu', action='store_true', 
                       help = 'Flag to indicate the use of gpu')
    parsed_args = parser.parse_args()
    
    args = vars(parsed_args)

    return args

def load_model(checkpoint_path, gpu):
    
    checkpoint = torch.load(checkpoint_path)
    hidden_units = checkpoint["hidden_units"]
    arch = checkpoint["arch"] 
    
    if arch == 'vgg11':
        model = models.vgg11(pretrained=False)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=False)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=False)

    
    classifier = nn.Sequential(OrderedDict([
#                          ('fc1', nn.Linear(25088, 256)),
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
#                          ('fc2', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    if(gpu == True):
        model.cuda()
    
    
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model

def process_image(image):  
    size = 256, 256
    
    new_side = 224
    center = 256 / 2
    start = 0
    
    im = Image.open(image)
    im.thumbnail(size)
    im = im.crop((start, start, new_side, new_side))
    np_image = np.array(im) / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image

def predict(image_path, model, topk):
    im = process_image(image_path)
    input = torch.FloatTensor(im).cuda()
    input.unsqueeze_(0)
    output = model.forward(input)
    ps = F.softmax(output, dim=1)
                          
    return torch.topk(ps, topk)

def get_cat_to_name(cat_to_name_path):
    cat_to_name = []
    
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def main():
    args = get_args()
    model = load_model(args["checkpoint"], args["gpu"])

    probs, classes = predict(args["input"], model, args["topk"])

    print("Most likely classes: \n")
    
    if(args["category_names"] == None):
        print(classes[0])
    else:    
        cat_to_name = get_cat_to_name(args["category_names"])
        
        for c in classes[0]:  
            for m in model.class_to_idx:
                if model.class_to_idx[m] == c:
                    print("* " + cat_to_name[m])
                
if __name__ == "__main__":
    main()