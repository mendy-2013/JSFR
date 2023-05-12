# -*- coding: utf-8 -*-

import os.path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image
import json

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

feature_path = 'Resnet34.jsnol'


def extractor(img_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]
    )

    img = Image.open(img_path)
    img = transform(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.view(1, -1)
    y = y.data.numpy().tolist()
    return y


if __name__ == '__main__':

    files_list = []

    image_list = "Flickr30k_image_list.txt"
    Flickr30k_path = "flickr30k-images"
    resnet34 = models.resnet34(pretrained=True)
    modules = list(resnet34.children())[:-2]  # delete the last fc layer and avgpooling.
    resnet34 = nn.Sequential(*modules)  # 从list转model
    for param in resnet34.parameters():
        param.requires_grad = False

    use_gpu = True

    dict = {}
    with open(image_list) as f:
        for line in f:
            image_name = line.split()[0]
            image_path = os.path.join(Flickr30k_path, image_name)
            image_feature = extractor(image_path, resnet34, use_gpu)
            print(image_path + " successfully extracted!")
            dict[image_name] = image_feature

    with open(feature_path, "w") as fw:
        jsobj = json.dump(dict, fw)
        fw.write(jsobj)
        fw.close()



