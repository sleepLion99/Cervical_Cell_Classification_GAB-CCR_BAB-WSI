import dgl
import os
from pathlib import Path
import torch
from torchvision import models, transforms, datasets
from data_utils import read_splited_data, MyDataSet
from torch.utils.data import DataLoader, Subset
from get_features_swin import extractor_img_features_dataloader
import numpy as np
import torch.nn.functional as F
from models import CRPRNet
import tqdm
from PIL import Image
import random

# 融入 patch 信息进行分类
# edit by lichaowei in 2023.04.20

# 存放 wsi 图片的路径
wsi_root_path = '/repository7504/AbnormalObjectDetetionDataset/01_labeled_wsi/20220419_new_data/small_small_wsi_gnn_data'
input_size = 224
batch_size = 1
device = torch.device("cuda:1")
num_workers=2
num_classes = 5
thredthod = 0.5

# path = '/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/20220419_new_data/wsi_patch/JF--ASC-H--3_25_QZ__ZZ_2020-02-21_17_56_52_HZ/JF--ASC-H--3_25_QZ__ZZ_2020-02-21_17_56_52_HZ^x_11264y_5120.jpg'
checkpoint_path = '/home/lichaowei/deeplearning/classification/GNN/duibishiyan/实验/swin_large/weights/best_0.78.pth'
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
    transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

patch_transforms = {
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

def read_img(path):
    img = Image.open(path)
    img = patch_transforms['train'](img)
    img = img.unsqueeze(dim=0)
    return img

def read_patch_img(wsi_patch_path, is_train=True):
    imgs = []
    for path in os.listdir(wsi_patch_path):
        img = Image.open(os.path.join(wsi_patch_path ,path))
        if is_train:
            img = patch_transforms['train'](img)
        else:
            img = patch_transforms['val'](img)
        img = img.unsqueeze(dim=0)
        imgs.append(img)
    return imgs

def evaluate(model, graph, features, labels, mask, patch_img1, patch_img2):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features, features, patch_img1, patch_img2)
        logits = logits[mask]
        labels = labels[mask].squeeze()
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels), correct.item(), len(labels)


workplace = '/home75/lichaowei/deeplearning/classification/GNN/wsi_small_0.78_patch_0420'
patch_save_path = '/home75/lichaowei/deeplearning/classification/GNN/wsi_small_patch_0420'
log_file_path = '/home75/lichaowei/deeplearning/classification/GNN/timm_gnn/chaocan/multihead=2.log'
write_file = open(log_file_path, 'a+')

if __name__ == '__main__':
    model = CRPRNet(1024, num_classes, 0.01)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=0.0001, weight_decay=5E-2)
    wsi_paths = os.listdir(wsi_root_path)
    epochs = 100
    best_acc = 0.0
    for epoch in range(epochs):
        # model train
        for wsi_path in wsi_paths:
            print(wsi_path)
            # 这块已经提取好了特征, 预处理图的信息
            absolute_wsi_path = os.path.join(wsi_root_path, wsi_path)
            train_images_path, train_images_label, val_images_path, val_images_label = read_splited_data(absolute_wsi_path, False)
            train_dataset = MyDataSet(train_images_path, train_images_label, transform=data_transforms['train'])
            val_dataset = MyDataSet(val_images_path, val_images_label, transform=data_transforms['val'])
            train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_dataloder = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            feature_save_path = os.path.join(workplace, wsi_path, "features.npy")
            label_save_path = os.path.join(workplace, wsi_path, "labels.npy")
            graph_save_path = os.path.join(workplace, wsi_path, "graph.dgl")
            location_save_path = os.path.join(workplace, wsi_path, "locations.npy")
            features = np.load(feature_save_path)
            labels = np.load(label_save_path)
            locations = np.load(location_save_path)
            graphs, _ = dgl.load_graphs(f'{graph_save_path}')
            g = graphs[0]
            num_train = int(len(train_dataloder))
            num_val = int(len(val_dataloder))
            train_idx = torch.arange(num_train)
            val_idx = torch.arange(num_train, num_train + num_val)
            num_nodes = features.shape[0]
            train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
            val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
            train_mask[:len(train_idx)] = True
            val_mask[len(train_idx):] = True
            node_labels = labels.squeeze()
            g = g.to(device)
            node_labels = g.ndata['label'].squeeze()
            node_features = g.ndata['feat']
            n_labels = num_classes
            # 预处理 patch 信息
            # patch_img = read_img(path).to(device)
            patch_imgs = read_patch_img(os.path.join(patch_save_path, wsi_path), is_train=True)
            random_index = random.sample(range(0, len(patch_imgs)), 2)
            patch_img1 = patch_imgs[random_index[0]].to(device)
            # patch_img2 = patch_imgs[random_index[1]].to(device)
            patch_img2 = None
            # 模型训练
            model.train()
            opt.zero_grad()
            logits = model(g, node_features, node_features, patch_img1, patch_img2)
            loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
            loss.backward()
            opt.step()
            acc, current_num, val_num = evaluate(model, g, node_features, node_labels, val_mask, patch_img1, patch_img2)
            # write_file.write(f'Epoch: {epoch}, current {wsi_path} val acc = {acc}\n')
            print(f"Epoch: {epoch}, current {wsi_path} val acc = {acc}")
        # model eval
        true_num = 0.0
        all_num = 0.0
        for wsi_path in wsi_paths:
            # 这块已经提取好了特征, 预处理图的信息
            absolute_wsi_path = os.path.join(wsi_root_path, wsi_path)
            train_images_path, train_images_label, val_images_path, val_images_label = read_splited_data(absolute_wsi_path, False)
            train_dataset = MyDataSet(train_images_path, train_images_label, transform=data_transforms['train'])
            val_dataset = MyDataSet(val_images_path, val_images_label, transform=data_transforms['val'])
            train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_dataloder = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            feature_save_path = os.path.join(workplace, wsi_path, "features.npy")
            label_save_path = os.path.join(workplace, wsi_path, "labels.npy")
            graph_save_path = os.path.join(workplace, wsi_path, "graph.dgl")
            location_save_path = os.path.join(workplace, wsi_path, "locations.npy")
            features = np.load(feature_save_path)
            labels = np.load(label_save_path)
            locations = np.load(location_save_path)
            graphs, _ = dgl.load_graphs(f'{graph_save_path}')
            g = graphs[0]
            num_train = int(len(train_dataloder))
            num_val = int(len(val_dataloder))
            train_idx = torch.arange(num_train)
            val_idx = torch.arange(num_train, num_train + num_val)
            num_nodes = features.shape[0]
            train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
            val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
            train_mask[:len(train_idx)] = True
            val_mask[len(train_idx):] = True
            node_labels = labels.squeeze()
            g = g.to(device)
            node_labels = g.ndata['label'].squeeze()
            node_features = g.ndata['feat']
            n_labels = num_classes
            # 预处理 patch 信息
            patch_imgs = read_patch_img(os.path.join(patch_save_path, wsi_path), is_train=False)
            random_index = random.sample(range(0, len(patch_imgs)), 2)
            patch_img1 = patch_imgs[random_index[0]].to(device)
            # patch_img2 = patch_imgs[random_index[1]].to(device)
            patch_img2 = None
            # 模型训练
            acc, current_num, val_num = evaluate(model, g, node_features, node_labels, val_mask, patch_img1, patch_img2)
            true_num += current_num
            all_num += val_num
        now_acc = true_num/all_num
        if now_acc>best_acc:
            best_acc = now_acc
            torch.save(model.state_dict(), f'/home75/lichaowei/deeplearning/classification/GNN/timm_gnn/chaocan/multihead=2_best.pth')
        torch.save(model.state_dict(), f'/home75/lichaowei/deeplearning/classification/GNN/timm_gnn/chaocan/multihead=2_last.pth')
        print(f"Epoch: {epoch}, current total val acc = {now_acc}, best_acc = {best_acc}")
        write_file.write(f'Epoch: {epoch}, current {wsi_path} val acc = {acc}\n')
        write_file.write(f"Epoch: {epoch}, current total val acc = {now_acc}, best_acc = {best_acc}")
        write_file.flush()
    write_file.close()




        