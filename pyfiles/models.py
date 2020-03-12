import torch
import torchvision
import numpy as np


class iCaRL(torch.nn.Module):
    def __init__(self, memory_size, device, feature_size, num_classes, lr):
        super().__init__()
        self.FeatureExtractor = torchvision.models.resnet34().to(device)
        self.fc_net = torch.nn.Linear(feature_size, num_classes).to(device)
        self.CELoss = torch.nn.CrossEntropyLoss().to(device)
        self.BCELoss = torch.nn.BCELoss().to(device)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = device
        self.total_classes = 0
        self.total_memory_size = memory_size
        self.exemplar_set = [torch.zeros(0).to(device), torch.zeros(0).to(device)] #[data, label]


    def forward(self, x):
        _x = self.FeatureExtractor(x)
        _x = self.fc_net(_x)
        return torch.nn.functional.sigmoid(_x)


    def classify(self, x):
        mu = torch.zeros(self.exemplar_set.shape[0]).to(self.device)
        phi_x = self.FeatureExtractor(x)
        for i, exemplar in enumerate(self.exemplar_set):
            mu[i] = self.FeatureExtractor(exemplar).sum(dim=0).mean(dim=0)

        return torch.min(((phi_x - mu) ** 2).sum(dim=0).sqrt())[1]


    def train(self, TrainDataLoader, num_new_class):
        """
        Method for training model

        Parameters
        ---------
        TrainDataLoader: new tasks' traindataloader\n
        num_new_class: # of new classes
        """
        self.UpdateRepresentation(TrainDataLoader)
        self.total_classes += num_new_class
        memory_size = self.total_memory_size / self.total_classes

        self.__ReduceExemplarSet(memory_size)
        self.__ConstructExemplarSet()


    def __UpdateRepresentation(self, TrainDataLoader):
        distillation_loss = 0.0
        classification_loss = 0.0

        ### Data Augmentation
        domain = self.__DataAugmentation(self.exemplar_set, TrainDataLoader)

        ### get q
        q = torch.zeros(0).to(self.device)
        for exemplar, _ in domain:
            q_i = forward(exemplar)
            q = torch.cat((q, q_i))

        ### loss with new data
        self.optim.zero_grad()
        for (x, y) in TrainDataLoader:
            x = x.to(self.device)
            y = y.to(self.device)

            out = self.forward(x)
            classification_loss += self.CELoss(out, y)
            distillation_loss += self.BCELoss(out, q)


        ### distillation loss

        classification_loss = self.CELoss()
        distillation_loss = self.BCELoss()

        loss = classification_loss + distillation_loss
        loss.backward()
        optim.step()


    def __DataAugmentation(self, target, TrainDataLoader):
        """
        Parameters
        ----------
        target: dataset to augment\n
        TrainDataLoader: new dataset
        """
        _target = [target[0].detach().clone(), target[1].detach().clone()]
        for (x, y) in TrainDataLoader:
            x = x.to(self.device)
            y = y.to(self.device)

            _target[0] = torch.cat((_target[0], x))
            _target[1] = torch.cat((_target[1], y))

        return _target

    def __IncrementWeight(self, num_classes):
        in_features = self.fc_net.in_features
        out_features = self.fc_net.out_features
        weights = self.fc_net.weight.data

        self.fc_net = torch.nn.Linear(in_features, num_classes).to(self.device)
        self.fc_net.weight.data[:out_features] = weights


    def __ConstructExemplarSet(self, TrainDataLoader):
        x_ = fv = P = fv_p =  torch.zeros(0)
        for (x, _) in TrainDataLoader:
            x = x.to(self.device)
            x_ = torch.cat((x_, x.unsqueeze(0)))
            num_data = x.shape[0]
            
            phi = self.FeatureExtractor(x)
            fv = torch.cat((fv, phi.unsqueeze(0)))

        mu = fv.mean(dim=0)
        memory_size = self.total_memory_size / self.total_classes
        
        for k in range(1, memory_size+1):
            pre_feature_sum = fv_p.sum(dim=0)
            features = fv.detach().clone() + pre_feature_sum
            
            index = torch.min(((mu - (features / k)) ** 2).sum(dim=0).sqrt())[1]
            P = torch.cat((P, x_[index].unsqueeze(0)))
            fv_p = torch.cat((fv_p, fv[index].unsqueeze(0)))

        self.exemplar_set = torch.cat((self.exemplar_set, P.unsqueeze(0)))


    def __ReduceExemplarSet(self, m):
        for exemplar in self.exemplar_set:
            exemplar = exemplar[:self.memory_size]


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        

    def forward(self, x):
        pass