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
entire_class = 20
class_per_task = 10
num_tasks = int(entire_class / class_per_task)

### Load iCIFAR-100 Dataset
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
TrainDatasets = []
TestDatasets = []

TrainDataLoaders = []
TestDataLoaders = []
for i in range(num_tasks):
    TrainDataSet = datasets.CIFAR100_IncrementalDataset(root='data/',
                                                        train=True,
                                                        transform=transform,
                                                        download=True,
                                                        classes=range(i * num_tasks, (i+1) * num_tasks))
    TrainDatasets.append(TrainDataSet)

    TestDataSet = datasets.CIFAR100_IncrementalDataset(root='data/', 
                                                        train=False,
                                                        transform=transform,
                                                        download=True,
                                                        classes=range(i * num_tasks, (i+1) * num_tasks))
    TrainDatasets.append(TestDataSet)
    '''
    TrainDataLoaders.append(torch.utils.data.DataLoader(TrainDataSet,
                                                        batch_size=batch_size,
                                                        num_workers=2,
                                                        shuffle=True))
    
    TestDataLoaders.append(torch.utils.data.DataLoader(TestDataSet, 
                                                       batch_size=batch_size,
                                                       num_workers=2,
                                                       shuffle=False))

    '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.iCaRL(memory_size=K, device=device, feature_size=2048, num_classes=class_per_task, lr=1e-3)
#net.to(device)
for i, TrainDataSet in enumerate(TrainDatasets):
    ### Training
    net.train(TrainDataSet, class_per_task, batch_size)

    #### Evaluation
    for j, TestDataLoader in enumerate(TestDataLoaders):
        if j >= i:
            break
        for (x, _) in TestDataLoader:
            x.to(device)
            net.classify(x)