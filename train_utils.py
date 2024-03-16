'''
Descripttion: tensorboard的简单使用
version: 1.0
Author: Geek
Date: 2022-08-23 21:59:57
LastEditors: Geek
LastEditTime: 2022-09-16 16:15:28
'''
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data_utils import MyDataSet, read_split_data, plot_class_preds
from torchvision import transforms
from tqdm import tqdm
import os
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import math
import os
import torch.nn.functional as F
from focal_loss import WeightedFocalLoss, MultiClassFocalLossWithAlpha
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

tags = ["train_loss", "train_acc", "val_acc", "learning_rate"]

data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.RandomRotation(30),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def train_parallel_net(net, train_iter, test_iter, epochs, lr,train_record_name, test_record_name,device_ids):
    """多GPU训练"""
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir="./logs")
    # 多GPU分布式训练
    net = nn.DataParallel(net, device_ids=device_ids)
    net.to(device_ids[0])
    # 需要将模型和数据放在一块指定的GPU上[主从并行模式], 网络在前向传播的时候会将数据分到各个显卡上
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    # 定义优化函数
    #  lr=lr, momentum=0.9, weight_decay=0.005
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    # 更新学习率方式 - 余弦退火
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.1) + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(trainer, lr_lambda=lf)
    # 初始化文件夹
    if not os.path.exists('./train_record'):
        os.mkdir('./train_record')
    if not os.path.exists('./test_record'):
        os.mkdir('./test_record')
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    # 训练
    best_val_acc = 0
    for epoch in range(epochs):
        net.train()
        for X, Y in tqdm(train_iter):
            X, Y = X.to(device_ids[0]), Y.to(device_ids[0])
            y_hat = net(X)
            l = loss(y_hat, Y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        scheduler.step()
        train_max_acc, train_mean_acc, train_mean_loss = acc_parallel(net, train_iter, loss, device_ids, False)
        train_str = f'EPOCH:{epoch+1}\tTRAIN_ACC: {train_mean_acc}\tTRAIN_LOSS:{train_mean_loss}\tTRAIN_MAX_ACC: {train_max_acc}\n'
        print(train_str)
        with open('./train_record/'+train_record_name, 'a') as file:
            file.write(train_str)
        test_max_acc, test_mean_acc, test_mean_loss = acc_parallel(net, test_iter, loss, device_ids, True)
        test_str = f'EPOCH:{epoch+1}\tTEST_ACC: {test_mean_acc}\tTEST_LOSS:{test_mean_loss}\tTEST_MAX_ACC: {test_max_acc}\n'
        with open('./test_record/'+test_record_name, 'a') as file:
            file.write(test_str)
        print(test_str)

        # 保存最优模型
        if test_mean_acc > best_val_acc:
            best_val_acc = test_mean_acc
            torch.save(net.module.state_dict(), f'./weights/best.pth')
        # 保存最后训练的模型
        torch.save(net.module.state_dict(), f'./weights/last.pth')

        # train_loss
        tb_writer.add_scalar(tags[0], train_mean_loss, epoch)
        # train_acc
        tb_writer.add_scalar(tags[1], train_mean_acc, epoch)
        # val_acc
        tb_writer.add_scalar(tags[2], test_mean_acc, epoch)
        # lr
        tb_writer.add_scalar(tags[3], lr, epoch)
        # 画预测图像
        fig = plot_gram_cam_class_preds(net, '/home/lichaowei/deeplearning/datasets/pred_cell_7', data_transforms['val'], 7, device_ids[0])
        if fig is not None:
            tb_writer.add_figure("predictions vs. actuals",
                                figure=fig,
                                global_step=epoch)

def acc_parallel(net, now_iter, loss, device_ids, is_Train = False):
    """准确率"""
    net.eval()
    max_acc = 0.0
    sum_acc = 0.0
    running_loss = 0.0
    total = 0
    now_iter = (tqdm(now_iter) if is_Train else now_iter)
    with torch.no_grad():
        for X, Y in now_iter:
            X, Y = X.to(device_ids[0]), Y.to(device_ids[0])
            y_hat = net(X)
            running_loss += loss(y_hat, Y)
            _,preds = torch.max(y_hat, 1)
            total_acc = (preds == Y).sum().item()
            now_len = len(Y)
            now_acc = total_acc / now_len
            sum_acc += now_acc
            if now_acc>max_acc:
                max_acc = now_acc
            total += 1
    mean_acc = sum_acc / total
    mean_loss = running_loss / total
    return max_acc, mean_acc, mean_loss  





"""
    单GPU训练
"""
def train_net(net, train_iter, test_iter, epochs, lr,train_record_name, test_record_name, device, is_focal_loss=True):
    """训练模型"""
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir="./logs")
    # 初始化权重
    net.apply(init_weight)
    net.to(device)
    # 定义损失函数
    if(is_focal_loss):
        # 根据数据集定制 alpha 的值
        # 这块暂时不将 reduction 设置为 mean, 因为发现使用 mean, loss很小, 收敛不了
        loss = MultiClassFocalLossWithAlpha(alpha=[0.02, 0.05, 0.13, 0.4, 0.4], gamma=2, reduction="sum", DEVICE=device)
    else:
        loss = nn.CrossEntropyLoss()
    # 定义优化函数
    # trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    pg = [p for p in net.parameters() if p.requires_grad]
    trainer = optim.AdamW(pg, lr=lr, weight_decay=5E-2)
    # 更新学习率方式 - 余弦退火
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.1) + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(trainer, lr_lambda=lf)
    # 初始化文件夹
    if not os.path.exists('./train_record'):
        os.mkdir('./train_record')
    if not os.path.exists('./test_record'):
        os.mkdir('./test_record')
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    # 训练
    is_first = False
    best_val_acc = 0
    for epoch in range(epochs):
        net.train()
        for X, Y, _ in tqdm(train_iter):
            X, Y = X.to(device), Y.to(device)
            if  not is_first:
                tb_writer.add_graph(net, X)
                is_first = True
            y_hat = net(X)
            l = loss(y_hat, Y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        scheduler.step()
        train_max_acc, train_mean_acc, train_mean_loss = acc(net, train_iter, loss, device, False)
        train_str = f'EPOCH:{epoch+1}\tTRAIN_ACC: {train_mean_acc}\tTRAIN_LOSS:{train_mean_loss}\tTRAIN_MAX_ACC: {train_max_acc}\n'
        print(train_str)
        with open('./train_record/'+train_record_name, 'a') as file:
            file.write(train_str)
        test_max_acc, test_mean_acc, test_mean_loss = acc(net, test_iter, loss, device, True)
        test_str = f'EPOCH:{epoch+1}\tTEST_ACC: {test_mean_acc}\tTEST_LOSS:{test_mean_loss}\tTEST_MAX_ACC: {test_max_acc}\n'
        with open('./test_record/'+test_record_name, 'a') as file:
            file.write(test_str)
        print(test_str)

        # 保存最优模型
        if test_mean_acc > best_val_acc:
            best_val_acc = test_mean_acc
            torch.save(net.state_dict(), f'./weights/best.pth')
        # 保存最后训练的模型
        torch.save(net.state_dict(), f'./weights/last.pth')

        # train_loss
        tb_writer.add_scalar(tags[0], train_mean_loss, epoch)
        # train_acc
        tb_writer.add_scalar(tags[1], train_mean_acc, epoch)
        # val_acc
        tb_writer.add_scalar(tags[2], test_mean_acc, epoch)
        # lr
        tb_writer.add_scalar(tags[3], lr, epoch)
        # 画预测图像
        # fig = plot_gram_cam_class_preds(net, '/home/lichaowei/deeplearning/datasets/pred_flower_photos', data_transform['val'], 5, device)
        # if fig is not None:
        
        #     tb_writer.add_figure("predictions vs. actuals",
        #                         figure=fig,
        #                         global_step=epoch)

def acc(net, now_iter, loss, device=0, is_Train = False):
    """准确率"""
    net.eval()
    net.to(device)
    max_acc = 0.0
    sum_acc = 0.0
    running_loss = 0.0
    total = 0
    now_iter = (tqdm(now_iter) if is_Train else now_iter)
    with torch.no_grad():
        for X, Y, _ in now_iter:
            X, Y = X.to(device), Y.to(device)
            y_hat = net(X)
            running_loss += loss(y_hat, Y)
            _,preds = torch.max(y_hat, 1)
            total_acc = (preds == Y).sum().item()
            now_len = len(Y)
            now_acc = total_acc / now_len
            sum_acc += now_acc
            if now_acc>max_acc:
                max_acc = now_acc
            total += 1
    mean_acc = sum_acc / total
    mean_loss = running_loss / total
    return max_acc, mean_acc, mean_loss

def init_weight(m):
    """初始化权重"""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    root_path = '/home/lichaowei/deeplearning/datasets/flower_photos'
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root_path)
    train_dataset = MyDataSet(train_images_path, train_images_label, transform=data_transform['train'])
    val_dataset = MyDataSet(val_images_path, val_images_label, transform=data_transform['val'])
    train_dataloder = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
    val_dataloder = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    # train_net(get_model(5), train_dataloder, val_dataloder, 200, 0.01, 'train_record.txt', 'test_record.txt', 0)
    # 查看 tensorboard 内容, tensorboard --logdir='/home/lichaowei/deeplearning/小工具/logs'