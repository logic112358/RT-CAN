from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

__all__ = ["JointLoss", "WeightedLoss"]


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0, third_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)
        # self.third = WeightedLoss(second, third_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)
    

class MRLoss(_Loss):
    """
    隐式对齐
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        rgb_4 = input[0][:, 0:2048, :]
        thermal_4 = input[0][:, 2048:, :]

        rgb_3 = input[1][:, 0:1024, :]
        thermal_3 = input[1][:, 1024:2048, :]

        rgb_2 = input[2][:, 0:512, :]
        thermal_2 = input[2][:, 512:1024, :]

        rgb_1 = input[3][:, 0:256, :]
        thermal_1 = input[3][:, 256:512, :]

        loss1 = F.l1_loss(rgb_4, thermal_4)
        loss2 = F.l1_loss(rgb_3, thermal_3) 
        loss3 = F.l1_loss(rgb_2, thermal_2)
        loss4 = F.l1_loss(rgb_1, thermal_1)
        return 0.4*loss1 + 0.3*loss2 + 0.2*loss3 + 0.1*loss4
