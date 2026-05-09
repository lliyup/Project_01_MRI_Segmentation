import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self,smooth = 1e-5,include_background = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self,logits,targets):
        num_classes = logits.shape[1]


        #logits->probabilities
        probs = torch.softmax(logits,dim = 1)


        #one-hot encoding
        targets_one_hot = F.one_hot(targets.long(),num_classes = num_classes)
        targets_one_hot = targets_one_hot.permute(0,4,1,2,3).float()

        if not self.include_background:
            probs = probs[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]

        dims  =(0,2,3,4)

        #计算交集和并集
        # 交集
        # 计算每个类别的交集
        intersection = torch.sum(probs*targets_one_hot,dim = dims)
        #  并集
        pred_sum = torch.sum(probs,dim = dims)
        targets_sum = torch.sum(targets_one_hot,dim = dims)
        union = pred_sum + targets_sum

        #计算Dice
        dice = (2*intersection + self.smooth)/(union + self.smooth)
        #平均Dice Loss
        loss = 1 - dice.mean()
        return loss
    
if __name__ == "__main__":
    loss_fn = DiceLoss(include_background = False)
    logits = torch.randn(2,2,32,32,32)
    targets = torch.randint(0,2,(2,32,32,32))

    loss = loss_fn(logits,targets)

    print("=" * 50)
    print("Dice Loss Test")
    print("=" * 50)
    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)
    print("Loss:", loss.item())
    print("=" * 50)