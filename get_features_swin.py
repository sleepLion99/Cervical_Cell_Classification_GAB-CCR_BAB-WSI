import timm
import torch
import numpy as np
from tqdm import tqdm
import os
model_name = 'swin_large_patch4_window7_224'

def get_model(num_classes, checkpoint_path):
    model = timm.create_model(model_name, pretrained=False, num_classes=5, checkpoint_path=checkpoint_path)
    return model

def extractor_img_features_dataloader(num_classes, checkpoint_path, device, my_dataloader, f_save_path, l_save_path, location_save_path):
    features_list = []
    label_list = []
    location_list = []
    if(os.path.exists(f_save_path) and os.path.exists(l_save_path)):
        features_list = np.load(f_save_path)
        label_list = np.load(l_save_path)
    else:
        model = get_model(num_classes, checkpoint_path).to(device)
        model.reset_classifier(0)
        model.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(output_size=1024),
            torch.nn.Identity()
        )
        print(model)
        model.to(device)
        for image, label, location in tqdm(my_dataloader):
            image = image.to(device)
            features = model(image)
            features_list.append(features.flatten().detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            ans_location = []
            for l in location:
                ans_location.append(l.item())
            # location = (l.item() for l in location)
            location_list.append(ans_location)
            # print(location_list)
        np.save(f_save_path, features_list)
        np.save(l_save_path, label_list)
        np.save(location_save_path, location_list)
    return features_list, label_list, location_list


if __name__ == '__main__':
    checkpoint_path = '/home/lichaowei/deeplearning/classification/GNN/timm_gnn/workplace_noloss_adam/weights/best.pth'
    extractor_img_features_dataloader(5, checkpoint_path, 0, "/home/lichaowei/deeplearning/classification/GNN/timm_gnn/workplace_noloss_adam/weights/best", "/home/lichaowei/deeplearning/classification/GNN/timm_gnn/workplace_noloss_adam/weights/best", "/home/lichaowei/deeplearning/classification/GNN/timm_gnn/workplace_noloss_adam/weights/best", "/home/lichaowei/deeplearning/classification/GNN/timm_gnn/workplace_noloss_adam/weights/best")