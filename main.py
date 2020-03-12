import torch
import torchvision
import matplotlib.pyplot as plt

import pyfiles.datasets as datasets
import pyfiles.models as models

batch_size = 64
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
TrainDataLoaders = []
TestDataLoaders = []
for i in range(10):
    # CIFAR100 Dataset
    TrainDataSet = CIFAR100_IncrementalDataset(root='data/',
                                               train=True,
                                               transform=transform,
                                               download=True,
                                               classes=range(i * 10, (i+1) * 10))
    
    TestDataSet = CIFAR100_IncrementalDataset(root='data/', 
                                              train=False,
                                              transform=transform,
                                              download=True,
                                              classes=range(i * 10, (i+1) * 10))
    
    TrainDataLoaders.append(torch.utils.data.DataLoader(TrainDataSet,
                                                        batch_size=batch_size,
                                                        shuffle=True))
    
    TestDataLoaders.append(torch.utils.data.DataLoader(TestDataSet, 
                                                       batch_size=batch_size,
                                                       shuffle=False))


K = 2000
epochs = 70
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.iCaRL(memory_size=K, device=device, feature_size=, num_classes=, lr=1e-3)
for TrainDataLoader in TrainDataLoaders:
    