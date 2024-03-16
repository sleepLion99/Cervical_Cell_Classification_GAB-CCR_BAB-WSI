import torch
from torch import nn
# from labml_helpers.module import Module
from dgl.nn.pytorch import GATv2Conv
import random


class GATV2(nn.Module):
    def __init__(self,
                in_feats,
                n_classes,
                dropout):
        super().__init__()
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # 1024 -> 512
        self.layers.append(GATv2Conv(in_feats, out_feats=128, num_heads=4, feat_drop=dropout))
        # 512 -> 256
        self.layers.append(GATv2Conv(128*4, 64, 4, feat_drop=dropout))
        # 256 -> 128
        self.layers.append(GATv2Conv(64*4, 32, 4, feat_drop=dropout))
        # 128 -> 64
        self.layers.append(GATv2Conv(32*4, 16, 4, feat_drop=dropout))
        # 64 -> 32
        self.layers.append(GATv2Conv(16*4, 8, 4, feat_drop=dropout))
        # 32 -> 5
        self.layers.append(GATv2Conv(8*4, n_classes, 1, feat_drop=dropout))

    def forward(self, g, x):
        y = x
        for l, layer in enumerate(self.layers):
            y = layer(g, y)
            y = y.reshape(y.shape[0], -1)
        return y

import timm

class CRPRNet(nn.Module):
    def __init__(self, in_feats, n_classes, dropout):
        super().__init__()
        self.n_classes = n_classes
        # 定义 gatv2 layers
        self.gatv2 = nn.ModuleList()
        # 1024 -> 512
        self.gatv2.append(GATv2Conv(in_feats, out_feats=128, num_heads=4, feat_drop=dropout))
        # 512 -> 256
        self.gatv2.append(GATv2Conv(128*4, 64, 2, feat_drop=dropout))

        self.gatv22 = nn.ModuleList()
        # 1024 -> 512
        self.gatv22.append(GATv2Conv(in_feats, out_feats=128, num_heads=4, feat_drop=dropout))
        # 512 -> 256
        self.gatv22.append(GATv2Conv(128*4, 64, 2, feat_drop=dropout))

        # 定义 patch 特征提取器 4*64 -> 256
        self.patch_generate = nn.ModuleList()
        self.patch_generate.append(timm.create_model(
            model_name="tf_efficientnetv2_m",
            pretrained=True,
            pretrained_cfg_overlay=dict(file='/home75/lichaowei/deeplearning/classification/GNN/timm_gnn/pytorch_model.bin'))
        )
        # self.patch_generate.append()
        # import torchvision.models as models
        # self.patch_generate.append(models.resnet50())
        self.patch_generate.append(nn.LeakyReLU(negative_slope=0.01))
        # 
        self.patch_generate.append(nn.Linear(1000, 128))

        # 定义 Transformer 结构
        # 修改头的数量
        self.trans = nn.MultiheadAttention(64*2, 2)

        # MLP 256 -> 5
        self.mlp = nn.ModuleList()
        # self.mlp.append(nn.Linear(512, 256))
        # self.mlp.append(nn.LeakyReLU(negative_slope=0.01))
        self.mlp.append(nn.Linear(128, 128))
        self.mlp.append(nn.LeakyReLU(negative_slope=0.01))
        self.mlp.append(nn.Linear(128, 64))
        self.mlp.append(nn.LeakyReLU(negative_slope=0.01))
        self.mlp.append(nn.Linear(64, 32))
        self.mlp.append(nn.LeakyReLU(negative_slope=0.01))
        self.mlp.append(nn.Linear(32, 5))

    def forward(self, g_diff, g_same, x1, patch_img1, patch_img2):
        # gatv2 提取特征
        gat_output1 = x1
        gat_output2 = x1
        for l, layer in enumerate(self.gatv2):
            gat_output1 = layer(g_diff, gat_output1)
            gat_output1 = gat_output1.reshape(gat_output1.shape[0], -1)
        for l, layer in enumerate(self.gatv22):
            gat_output2 = layer(g_same, gat_output2)
            gat_output2 = gat_output2.reshape(gat_output2.shape[0], -1)
        # patch 提取特征
        for l, layer in enumerate(self.patch_generate):
            patch_img1 = layer(patch_img1)
            patch_img2 = layer(patch_img2)

        gat_output = gat_output1+gat_output2
        # patch_input = torch.cat([patch_img1, patch_img2], dim=1)
        patch_input = patch_img1+patch_img2
        # transformer 连接 gatv2 和 patch 特征
        key = value = patch_input

        attn_output, attn_output_weights = self.trans(gat_output, key, value)

        # 残差连接
        gat_output = gat_output + attn_output
        # mlp
        for l, layer in enumerate(self.mlp):
            gat_output = layer(gat_output)
        return gat_output
