import torch


def dice_score(probs,targets,class_id = 1,smooth = 1e-5):
    #计算单类别Dice Score

    probs_mask = (probs == class_id).float()
    targets_mask = (targets == class_id).float()


    intersection = torch.sum(probs_mask*targets_mask)
    probs_sum = torch.sum(probs_mask)
    targets_sum = torch.sum(targets_mask)
    union_sum = probs_sum + targets_sum

    dice = (2*intersection + smooth)/(union_sum + smooth)

    return dice.item()

def iou_score(probs,targets,class_id = 1,smooth = 1e-5):
    #计算单类别IoU Score

    probs_mask = (probs == class_id).float()
    targets_mask = (targets == class_id).float()

    intersection = torch.sum(probs_mask*targets_mask)
    probs_sum = torch.sum(probs_mask)
    targets_sum = torch.sum(targets_mask)
    union_sum = probs_sum + targets_sum - intersection

    iou = (intersection + smooth)/(union_sum + smooth)

    return iou.item()

def logits_to_prediction(logits):

    prob = torch.argmax(logits,dim = 1)
    return prob


if __name__ == "__main__":
    logits = torch.randn(2,2,32,32,32)
    targets = torch.randint(0,2,(2,32,32,32))
    probs = logits_to_prediction(logits)

    dice = dice_score(probs,targets,class_id = 1)
    iou = iou_score(probs,targets,class_id = 1)

    print("=" * 50)
    print("Metrics Test")
    print("=" * 50)
    print("Prediction shape:", probs.shape)
    print("Target shape:", targets.shape)
    print("Dice:", dice)
    print("IoU:", iou)
    print("=" * 50)