
import os
import torch
import torchvision

class datasets(object):
    def __init__(self):
        self.dataset_list_plain = ['mnist']
        self.dataset_list = [('mnist', self.load_mnist)]
        self.data_path = './data'

    def create_dataset(self, dataset="mnist", batch_size=128, train_size=50000, valid_size=50000, data_aug=False, img_size=224):
        if dataset not in self.dataset_list_plain:
            raise ValueError('Dataset not available at the moment')
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_size = train_size
        self.valid_size = valid_size
        self.data_aug = data_aug
        self.img_size=img_size
        return self.reload()
    
    def reload(self):
        data = {mt[0] : mt[1] for mt in self.dataset_list}
        return data[self.dataset]()
    def get_transform(self, norm):
        if self.data_aug:
            self.transform = torchvision.transforms.Compose([
                            #torchvision.transforms.Resize(self.img_size),                
                            torchvision.transforms.RandomResizedCrop(self.img_size, ratio=(0.95, 1.05), scale=(0.40, 1.0)),
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(norm[0], norm[1]),                     
                            ])
        else:
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.CenterCrop(self.img_size),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(norm[0], norm[1]),
                            ])
    def load_mnist(self):

        # MNIST data calculated mean and std
        self.get_transform(((0.1307,), (0.3081,)))

        _test = torchvision.datasets.MNIST(self.data_path, train=False, download=True, transform=self.transform)

        _train = torchvision.datasets.MNIST(self.data_path, train=True, download=True, transform=self.transform)
        _train.train_data = _train.train_data[:self.train_size]
        _train.train_labels = _train.train_labels[:self.train_size]

        _valid = torchvision.datasets.MNIST(self.data_path, train=True, download=True, transform=self.transform)
        _valid.train_data = _valid.train_data[self.valid_size:]
        _valid.train_labels = _valid.train_labels[self.valid_size:]        

        loader = {
            'train': torch.utils.data.DataLoader(
                _train, batch_size=self.batch_size, shuffle=True,
                pin_memory=True, num_workers=10),
            'valid': torch.utils.data.DataLoader(
                _valid, batch_size=self.batch_size, shuffle=False),
            'test': torch.utils.data.DataLoader(
                _test, batch_size=self.batch_size, shuffle=False)}  
        return loader          