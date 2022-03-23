import torch.nn.functional as F
import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class TeethNetLoss(nn.Module):
    def __init__(self):
        super(TeethNetLoss, self).__init__()
        self.loss = BCEDiceLoss()

    def forward(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        loss0 = self.loss(d0, labels_v)
        loss1 = self.loss(d1, labels_v)
        loss2 = self.loss(d2, labels_v)
        loss3 = self.loss(d3, labels_v)
        loss4 = self.loss(d4, labels_v)
        loss5 = self.loss(d5, labels_v)
        loss6 = self.loss(d6, labels_v)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
            loss0.data, loss1.data, loss2.data, loss3.data, loss4.data, loss5.data, loss6.data,))
        return loss0, loss
