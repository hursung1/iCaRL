import torch
import torchvision
import numpy as np
from copy import deepcopy


class iCaRL(torch.nn.Module):
    def __init__(self, memory_size, device, feature_size, num_classes, lr):
        super().__init__()
        self.FeatureExtractor = torchvision.models.resnet34(num_classes=feature_size).to(device)
        self.fc_net = torch.nn.Linear(feature_size, num_classes).to(device)
        self.CELoss = torch.nn.CrossEntropyLoss().to(device)
        self.BCELoss = torch.nn.BCELoss().to(device)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = device
        self.total_classes = 0
        self.total_memory_size = memory_size
        self.exemplar_set = [torch.zeros(0).to(device), torch.zeros(0, dtype=torch.long).to(device)] #[data, label]


    def forward(self, x):
        _x = self.FeatureExtractor(x)
        _x = self.fc_net(_x)
        return torch.sigmoid(_x)


    def classify(self, x):
        mu = torch.zeros(self.exemplar_set.shape[0]).to(self.device)
        phi_x = self.FeatureExtractor(x)
        for i, exemplar in enumerate(self.exemplar_set):
            mu[i] = self.FeatureExtractor(exemplar).sum(dim=0).mean(dim=0)

        return torch.min(((phi_x - mu) ** 2).sum(dim=0).sqrt())[1]


    def train(self, TrainDataSet, num_new_class, batch_size):
        """
        Method for training model

        Parameters
        ---------
        TrainDataSet: new tasks' traindataset\n
        num_new_class: # of new classes
        """
        _TrainDataSet = deepcopy(TrainDataSet)
        self.batch_size = batch_size
        self.__UpdateRepresentation(_TrainDataSet, self.batch_size)
        self.total_classes += num_new_class
        memory_size = int(self.total_memory_size / self.total_classes)

        self.__ReduceExemplarSet(memory_size)
        self.__ConstructExemplarSet(TrainDataSet)


    def __UpdateRepresentation(self, TrainDataSet, batch_size):
        distillation_loss = classification_loss = loss = 0.0

        
        ### Data Augmentation
        self.__DataAugmentation(TrainDataSet, self.exemplar_set)
        TrainDataLoader = torch.utils.data.DataLoader(TrainDataSet,
                                                    batch_size=batch_size,
                                                    #num_workers=2,
                                                    shuffle=True)

        ### get q
        with torch.no_grad():
            q = torch.zeros(0).to(self.device)
            for exemplar, _ in TrainDataLoader:
                exemplar = exemplar.to(self.device)
                q_i = self.forward(exemplar)
                q = torch.cat((q, q_i))

        ### loss with new data
        self.optim.zero_grad()
        for i, (x, y) in enumerate(TrainDataLoader):
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.forward(x)
            loss = self.CELoss(out, y)
            #print(out.shape, q.shape)
            
            loss += self.BCELoss(out, q[batch_size*i : batch_size*(i+1)])
            loss.backward()
            self.optim.step()


    def __DataAugmentation(self, TrainDataSet, target):
        """
        Parameters
        ----------
        target: dataset to augment. (data, label) pair\n
        TrainDataLoader: new dataset
        """
        data, label = target
        if data.shape[0] > 0 and label.shape[0] > 0:
            data = data.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            #print(data.shape, label.shape)
            TrainDataSet.append(data, label)

    def __IncrementWeight(self, num_classes):
        in_features = self.fc_net.in_features
        out_features = self.fc_net.out_features
        weights = self.fc_net.weight.data

        self.fc_net = torch.nn.Linear(in_features, num_classes).to(self.device)
        self.fc_net.weight.data[:out_features] = weights


    def __ConstructExemplarSet(self, TrainDataSet):
        x_ = fv = P = fv_p =  torch.zeros(0).to(self.device)
        TrainDataLoader = torch.utils.data.DataLoader(TrainDataSet,
                                                    batch_size=self.batch_size,
                                                    #num_workers=2,
                                                    shuffle=False)
        for (x, _) in TrainDataLoader:
            x = x.to(self.device)
            x_ = torch.cat((x_, x))
            num_data = x.shape[0]
            
            phi = self.FeatureExtractor(x)
            fv = torch.cat((fv, phi.unsqueeze(0)))

        mu = fv.mean(dim=0)
        memory_size = int(self.total_memory_size / self.total_classes)
        
        for k in range(1, memory_size+1):
            pre_feature_sum = fv_p.sum(dim=0)
            features = fv.detach().clone() + pre_feature_sum
            
            index = torch.min(((mu - (features / k)) ** 2).sum(dim=0).sqrt())[1]
            P = torch.cat((P, x_[index].unsqueeze(0)))
            fv_p = torch.cat((fv_p, fv[index].unsqueeze(0)))

        self.exemplar_set = torch.cat((self.exemplar_set, P.unsqueeze(0)))


    def __ReduceExemplarSet(self, m):
        for exemplar in self.exemplar_set:
            exemplar = exemplar[:m]


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        

    def forward(self, x):
        pass