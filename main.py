import torch
import torchvision
import matplotlib.pyplot as plt

import pyfiles.datasets as datasets

batch_size = 64
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
TrainDataLoader = []
TestDataLoader = []
for i in range(5):
    TrainDataset = datasets.CIFAR100_IncrementalDataset(train=True, 
                                                        download=True, 
                                                        transform=transform,
                                                        classes=range(i*2, i*2 + 1))
    TestDataset = datasets.CIFAR100_IncrementalDataset(train=False,
                                                        download=True, 
                                                        transform=transform,
                                                        classes=range(i*2, i*2 + 1))
    TrainDataLoader.append(torch.utils.data.DataLoader(TrainDataset, 
                                                        batch_size=batch_size, 
                                                        shuffle=True,
                                                        num_workers=2))
    TestDataLoader.append(torch.utils.data.DataLoader(TestDataset, 
                                                        batch_size=batch_size, 
                                                        shuffle=True,
                                                        num_workers=2))


K = 2000
epochs = 70
