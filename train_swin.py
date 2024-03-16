import timm
import torch
from torchvision import datasets, models, transforms
from data_utils import read_splited_data, MyDataSet
from torch.utils.data import DataLoader
from train_utils import train_net, train_parallel_net

DEVICE = torch.device("cuda:1")
checkpoint_path = '/home/lichaowei/deeplearning/classification/GNN/timm_gnn/swin_large_patch4_window7_224_22kto1k.pth'
# 数据根目录
root_path = '/repository04/AbnormalObjectDetetionDataset/01_labeled_wsi/20220419_new_data/small_small_wsi_cnn_data'
lr = 0.0001
input_size = 224
num_classes = 5
model_name = 'swin_large_patch4_window7_224'
num_workers = 16
batch_size = 16
epoch = 100
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


if __name__ == '__main__':
    model = timm.create_model(model_name, pretrained=True)
    # model.aux_head = torch.nn.Linear(in_features=model.aux_head.in_features, out_features=num_classes)
    # model.norm = torch.nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=num_classes)
    train_images_path, train_images_label, val_images_path, val_images_label = read_splited_data(root_path, True)
    train_dataset = MyDataSet(train_images_path, train_images_label, transform=data_transforms['train'])
    val_dataset = MyDataSet(val_images_path, val_images_label, transform=data_transforms['val'])
    train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloder = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_net(
        model, train_dataloder, val_dataloder, epoch, lr, 
        'train_swin_large_small_24_p16_224_5classes.txt', 'test_swin_large_small_24_p16_224_5classes.txt', DEVICE, False)