# JSFU
The code of our paper: Joint Semantic Deep Learning Algorithm For Object Detection under Foggy Road Conditions

# The code is based on the Pytorch 1.7.1

# Pre-trained weights download address (Download it and put it in your backbone folder) :
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

# Our model weights download address (Download it and put it in your save_weight folder) :

# For training our model for Cityscapes and Foggy Cityscapes datasets:
  python train_res50_fpn_unet.py

# For testing out model:
  python validation.py

# For images on our model：
  python predict.py
