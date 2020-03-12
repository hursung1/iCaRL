import os
import torch
import torchvision
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def permute_mnist(num_task, batch_size):
    """
    Returns PermutedMNISTDataLoaders

    Parameters
    ------------
    num_task: number of tasks\n
    batch_size: size of minibatch

    Returns
    ------------
    num_task numbers of TrainDataLoader, TestDataLoader

    """
    train_loader = {}
    test_loader = {}
    
    train_data_num = 0
    test_data_num = 0
    
    for i in range(num_task):
        shuffle_seed = np.arange(28*28)
        np.random.shuffle(shuffle_seed)
        
        train_PMNIST_DataLoader = PermutedMNISTDataLoader(train=True, shuffle_seed=shuffle_seed)
        test_PMNIST_DataLoader = PermutedMNISTDataLoader(train=False, shuffle_seed=shuffle_seed)
        
        train_data_num += train_PMNIST_DataLoader.getNumData()
        test_data_num += test_PMNIST_DataLoader.getNumData()
        
        train_loader[i] = torch.utils.data.DataLoader(
                train_PMNIST_DataLoader,
                batch_size=batch_size)
        
        test_loader[i] = torch.utils.data.DataLoader(
                test_PMNIST_DataLoader,
                batch_size=batch_size)
    
    return train_loader, test_loader, int(train_data_num/num_task), int(test_data_num/num_task)


def imshow(img):
    #img = (img+1)/2    
    img = img.squeeze()
    np_img = img.numpy()
    print(np_img.shape)
    plt.imshow(np_img, cmap='gray')
    plt.show()

    
def imshow_grid(img):
    img = torchvision.utils.make_grid(img.cpu().detach())
    #img = (img+1)/2
    npimg = img.numpy()
    #npimg = np.transpose(img.numpy(), (1, 2, 0))
    print(npimg.shape)
    plt.imshow(npimg[0], cmap='gray')
    plt.show()


def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            torch.nn.init.xavier_normal_(p)
        else:
            torch.nn.init.uniform_(p, 0.1, 0.2)

