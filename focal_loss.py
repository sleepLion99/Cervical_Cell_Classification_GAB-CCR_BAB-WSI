import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class WeightedFocalLoss(nn.Module):
    """
        单类 FocalLoss
    """
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        cross_loss = nn.CrossEntropyLoss(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-cross_loss)
        F_loss = at*(1-pt)**self.gamma * cross_loss
        return F_loss.mean()

class MultiClassFocalLossWithAlpha(nn.Module):
    """
        多类 FocalLoss
    """
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean', DEVICE=torch.device("cuda:1")):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(DEVICE)
        self.gamma = torch.tensor(gamma).to(DEVICE)
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

class Focal_Loss(nn.Module):
	def __init__(self,weight,gamma=2):
		super(Focal_Loss,self).__init__()
		self.gamma=gamma
		self.weight=weight
    
	def forward(self,preds,labels):
		"""
		preds:softmax输出结果
		labels:真实值
		"""
		eps=1e-7
		y_pred =preds.view((preds.size()[0],preds.size()[1],-1)) #B*C*H*W->B*C*(H*W)
		print(y_pred)
		target=labels.view(y_pred.size()) #B*C*H*W->B*C*(H*W)
		
		ce=-1*torch.log(y_pred+eps)*target
		floss=torch.pow((1-y_pred),self.gamma)*ce
		floss=torch.mul(floss,self.weight)
		floss=torch.sum(floss,dim=1)
		return torch.mean(floss)


if __name__ == '__main__':
    # 测试 focal_loss 
    outputs = torch.tensor([[2, 1., 2, 3, 5],
                            [2.5, 1, 3, 4, .5]], device='cuda')
    targets = torch.tensor([0, 1], device='cuda')
    print(torch.nn.functional.softmax(outputs, dim=1))

    fl= MultiClassFocalLossWithAlpha([0.02, 0.05, 0.13, 0.4, 0.4], 2, DEVICE=torch.device("cuda:0"), reduction="sum")
    f2= MultiClassFocalLossWithAlpha([0.02, 0.05, 0.13, 0.4, 0.4], 2, DEVICE=torch.device("cuda:0"), reduction="mean")
    print(fl(outputs, targets))
    print(f2(outputs, targets))
    loss = nn.CrossEntropyLoss()
    print(loss(outputs, targets))