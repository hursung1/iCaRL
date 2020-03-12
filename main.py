import torch
import torchvision
import matplotlib.pyplot as plt

import pyfiles.datasets as datasets
import pyfiles.models as models

### Hyperparameters
K = 2000
epochs = 70
batch_size = 64
feature_size = 1
entire_class = 100
class_per_task = 10
num_tasks = entire_class / class_per_task

### Load iCIFAR-100 Dataset
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
TrainDataLoaders = []
TestDataLoaders = []
for i in range(num_tasks):
    TrainDataSet = CIFAR100_IncrementalDataset(root='data/',
                                               train=True,
                                               transform=transform,
                                               download=True,
                                               classes=range(i * num_tasks, (i+1) * num_tasks))
    
    TestDataSet = CIFAR100_IncrementalDataset(root='data/', 
                                              train=False,
                                              transform=transform,
                                              download=True,
                                              classes=range(i * num_tasks, (i+1) * num_tasks))
    
    TrainDataLoaders.append(torch.utils.data.DataLoader(TrainDataSet,
                                                        batch_size=batch_size,
                                                        shuffle=True))
    
    TestDataLoaders.append(torch.utils.data.DataLoader(TestDataSet, 
                                                       batch_size=batch_size,
                                                       shuffle=False))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.iCaRL(memory_size=K, device=device, feature_size=1, num_classes=class_per_task, lr=1e-3)
for i, TrainDataLoader in enumerate(TrainDataLoaders):
    net.train(TrainDataLoader, class_per_task)

    #### Evaluation
    for j, TestDataLoader in enumerate(TestDataLoaders):
        if j >= i:
            break
        