import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms, datasets
device = torch.device("cuda:0")
import timm
from PIL import Image

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
    transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

path = '/repository7504/AbnormalObjectDetetionDataset/01_labeled_wsi/20220419_new_data/wsi_patch/JF--ASC-H--3_25_QZ__ZZ_2020-02-21_17_56_52_HZ/JF--ASC-H--3_25_QZ__ZZ_2020-02-21_17_56_52_HZ^x_11264y_5120.jpg'

img = Image.open(path)
img = data_transforms['train'](img)
img = img.unsqueeze(dim=0)
print(img.shape)


x = torch.ones(1,3, 384, 384).to(device)
# from models import CRPRNet
# model = CRPRNet(1024, 5, 0.1)
# model.to(device)

# print(timm.list_models())


model = timm.create_model(
    model_name="tf_efficientnetv2_m",
    pretrained=True,
    pretrained_cfg_overlay=dict(file='/home75/lichaowei/deeplearning/classification/GNN/timm_gnn/pytorch_model.bin')).to(device)
model.global_pool = nn.Identity()
# model.global_pool = torch.nn.AdaptiveAvgPool1d(output_size=128)
# model.classifier = nn.Identity()
print(model)
print(model(x).shape)
# y = torch.ones(1,64)
# z = torch.cat((x, y), dim=1)
# print(z.shape)
