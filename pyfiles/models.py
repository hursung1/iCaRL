import torch
import torchvision
import numpy as np


class iCaRL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = torchvision.models.resnet34()
        self.fc_net = torch.nn.Linear()


    def classify(self):
        pass

    def train(self):
        pass

    def UpdateRepresentation(self):
        pass

    def ConstructExemplarSet(self):
        pass

    def ReduceExemplarSet(self):
        pass


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        

    def forward(self, x):
        pass