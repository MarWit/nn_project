from __future__ import print_function

import os
import torch
import torchvision
import scipy.io as sio
import numpy as np
import h5py

import errno
import os

import torch.utils.data as data
from PIL import Image

from lib import plotter, utils

import glob


class datasets(object):
    def __init__(self):
        self.dataset_list_plain = ['mnist', 'mnist_m', 'svhn', 'usps', 'pacs', 'fergd']
        self.dataset_list = list(zip(self.dataset_list_plain, [self.load_mnist, self.load_mnist_m, self.load_svhn, self.load_usps, self.load_pacs, self.load_fergd]))
        self.data_path = './data'    

    def create_dataset(self, dataset, train_size=None, data_aug=False, img_size=224, transform=None, pacs='art_painting', pacs_heuristic=False, extra=False, p=0.0):
        """Create dataset

        Parameters:
            dataset    : string
                Name of dataset, one of ['mnist', 'mnist_m', 'svhn', 'usps', 'pacs']
            train_size : int
                number of pictures to pick from
            data_aug   : bool
                turn on automatic image augmentation
            img_size   : int 
                size of single image returned in batch
            transform  : torchvision.transforms
                custom transform, data_aug needs to be set to True
            pacs       : string
                choose PACS data type - one of the following [art_painting, cartoon, photo, sketch]                
            pacs_heuristics : bool
                if set to true batch will contain mix of pacs types except the one from 'pacs' argument which will be in testing 

        You can directly access data via datasets._train, datasets._test and datasets._extra variables
        E.g datasets._train.train_data[index], datasets._extra.extra_labels[index]
        """
        if dataset not in self.dataset_list_plain:
            raise ValueError('Dataset not available at the moment')
        
        self.dataset = dataset
        self.train_size = train_size
        self.data_aug = data_aug
        self.img_size=img_size
        self.transform = transform
        self.pacs = pacs
        self.heu = pacs_heuristic
        self.ex = extra
        self.p = p

        if 'self._train' in locals():
            if self._train is not None:
                del(self._train)
        if 'self._test' in locals():
            if self._test is not None:
                del(self._test)
        if 'self._extra' in locals():
            if self._extra is not None:
                del(self._extra)                                

        self._train = None
        self._test = None
        self._extra = None
        
        data = {mt[0] : mt[1] for mt in self.dataset_list}
        return data[self.dataset]()

    def get_transform(self, norm):
        """Get transform
        
        Parameters:
            norm : tuple(3)
                norm to use by transforms.Normalize - [std, mean]
        
        Returns default transform used by dataset when transform param isn't specified
        When data_aug == True, it returns data augumentation variant instead

        """
        if self.data_aug:
            self.transform = torchvision.transforms.Compose([       
                            torchvision.transforms.RandomResizedCrop(self.img_size, ratio=(0.85, 1.05), scale=(0.40, 1.0)),
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.ToTensor(),
                            #torchvision.transforms.Normalize(norm[0], norm[1]),                     
                            ])
        else:
            self.transform = torchvision.transforms.Compose([
                             torchvision.transforms.Resize(self.img_size),
                             torchvision.transforms.ToTensor(),
                             #torchvision.transforms.Normalize(norm[0], norm[1]),
                            ])

    def load_mnist(self):
        # MNIST data calculated mean and std
        if self.transform is None:
            self.get_transform(((0.5,0.5), (0.5,0.5))) 

        self._train = torchvision.datasets.MNIST(self.data_path, train=True, download=True, transform=self.transform)
        self._test = torchvision.datasets.MNIST(self.data_path, train=False, download=True, transform=self.transform)

    def load_mnist_m(self):
        if self.transform is None:
            self.get_transform(((0.5,0.5,0.5), (0.5,0.5,0.5))) 

        self._train = MNISTM(self.data_path, transform=self.transform, train=True)
        self._test = MNISTM(self.data_path, transform=self.transform, train=False)

    def load_svhn(self):
        if self.transform is None:
            self.get_transform(((0.5,0.5,0.5), (0.5,0.5,0.5)))
        
        self._train = torchvision.datasets.SVHN(self.data_path + '/svhn', split='train', download=True, transform=self.transform)
        self._test = torchvision.datasets.SVHN(self.data_path + '/svhn', split='test', download=True, transform=self.transform)
        self._extra = torchvision.datasets.SVHN(self.data_path + '/svhn', split='extra', download=True, transform=self.transform)

    def load_usps(self):
        if self.transform is None:
            self.get_transform(((0.5,0.5,0.5), (0.5,0.5,0.5))) 

        self._train = USPS(self.data_path, transform=self.transform, train=True)
        self._test = USPS(self.data_path, transform=self.transform, train=False)
    def load_fergd(self):
        if self.transform is None:
            self.get_transform(((0.5,0.5,0.5), (0.5,0.5,0.5)))

        self._train = FERGD(self.data_path, transform=self.transform, split='train', p=self.p)
        self._test = FERGD(self.data_path, transform=self.transform, split='test', p=self.p)
        self._extra = FERGD(self.data_path, transform=self.transform, split='valid', p=self.p)
        


    def load_pacs(self):
        if self.transform is None:
            self.get_transform(((0.5,0.5,0.5), (0.5,0.5,0.5)))
        if self.heu is False:
            self._train = PACS(self.data_path, transform=self.transform, split='train', style=self.pacs)
            self._test = PACS(self.data_path, transform=self.transform, split='test', style=self.pacs)
            self._extra = PACS(self.data_path, transform=self.transform, split='validate', style=self.pacs)
        else:
            self._test = PACS(self.data_path, transform=self.transform, split='test', style=self.pacs)
            b = False
            stack_d = []
            stack_l = []

            for st in ['art_painting', 'cartoon', 'photo', 'sketch']:
                if st is not self.pacs:
                    if b is False:
                        self._train = PACS(self.data_path, transform=self.transform, split='train', style=st)
                        stack_d.append(self._train.train_data)
                        stack_l.append(self._train.train_labels)
                        if self.ex is True:
                            tmp = PACS(self.data_path, transform=self.transform, split='validate', style=st)                          
                            stack_d.append(tmp.extra_data)
                            stack_l.append(tmp.extra_labels)
                        b = True
                    else:
                        tmp = PACS(self.data_path, transform=self.transform, split='train', style=st)
                        stack_d.append(tmp.train_data)
                        stack_l.append(tmp.train_labels)
                        if self.ex is True:
                            tmp = PACS(self.data_path, transform=self.transform, split='validate', style=st)                            
                            stack_d.append(tmp.extra_data)
                            stack_l.append(tmp.extra_labels)
                                                    
            self._train.train_data = np.concatenate(stack_d)
            self._train.train_labels = np.concatenate(stack_l)




    def batch_loader(self, b_size):
        """ Batch loader

        Parameters:
            b_size : int
                specify size of single batch

        Returns iterable batch loaders, returning tuple (img, label)
        Pick loader via ['train'], ['test'] or ['extra'] if available
        """
        if self._extra is not None:
            loader = {
                'train': torch.utils.data.DataLoader(
                    self._train, batch_size=b_size, shuffle=True,
                    pin_memory=True, num_workers=10),
                'extra' : torch.utils.data.DataLoader(
                    self._extra, batch_size=b_size, shuffle=True,
                    pin_memory=True, num_workers=10),
                'test': torch.utils.data.DataLoader(
                    self._test, batch_size=b_size, shuffle=False,)
                    }
            return loader
        if self._train is not None and self._test is not None:            
            loader = {
                'train': torch.utils.data.DataLoader(
                    self._train, batch_size=b_size, shuffle=True,
                    pin_memory=True, num_workers=10),
                'test': torch.utils.data.DataLoader(
                    self._test, batch_size=b_size, shuffle=False)}  
            return loader                              
        raise ValueError('Init dataset first!')

    
    def item(self, index, split):
        """Item

        Parameters:
            index : int
                index of desired item
            split : string
                one of the following : 'test', 'train', 'extra' (if available)
                specify split of data to choose from

        Returns single value from database in a form of tuple (image, label)    
        """
        if self._train is None or self._test is None:
            raise ValueError('Init dataset first!')

        if split == 'train':
            return self._train.__getitem__(index)
        if split == 'test':
            return self._test.__getitem__(index)
        return self._extra.__getitem__(index)

    def length(self, split):
        """Length

        Parameters:
            split : string
                one of the following : 'test', 'train', 'extra' (if available)
                specify split of data to choose from
        Returns number of items in dataset
        """
        if self._train is None or self._test is None:
            raise ValueError('Init dataset first!')
                
        if split == 'train':
            return self._train.__len__()
        if split == 'test':
            return self._test.__len__()
        return self._extra.__len__()
    
    def plot_img(self, index, split):
        """Plot img

        Parameters:
            index : int
                index of element to plot
            split : string
                one of the following : 'test', 'train', 'extra' (if available)
                specify split of data to choose from            

        Plots choosen image
        """

        img, _ = self.item(index, split)
        plotter.plot_mat(img.numpy()[None, :])


class PACS(data.Dataset):
    url = "http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017"
    file = 'pacs/'
    styles =  ['art_painting', 'cartoon', 'photo', 'sketch']


    def __init__(self, root, transform=None, split='train', style=None):
        super(PACS, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.style = style
        if self.style is None or self.style not in self.styles:
            raise ValueError('Pick one of the following styles: ' + ','.join(self.styles))

        if not self._check_exists(): 
            raise ValueError('Dataset file not found, download at: ' + self.url)
        
        with h5py.File(os.path.join(os.path.join(self.root, self.file, (self.style + '_' + self.split + '.hdf5'))), 'r') as f:
            if self.split is 'train':
                self.train_data = (f.get('images')[:])
                self.train_labels = (f.get('labels')[:])
            if self.split is 'test':
                self.test_data = (f.get('images')[:])
                self.test_labels = (f.get('labels')[:])
            if self.split is 'validate':
                self.extra_data = (f.get('images')[:])
                self.extra_labels = (f.get('labels')[:])  

    def __getitem__(self, index):
        if self.split is 'train':
            img, target = self.train_data[index], self.train_labels[index]
        if self.split is 'test':
            img, target = self.test_data[index], self.test_labels[index]
        if self.split is 'validate':
            img, target = self.extra_data[index], self.extra_labels[index]

        img = img[:,:,::-1]
        img = Image.fromarray(np.int8(img), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.split is 'train':
            return len(self.train_data)
        if self.split is 'test':
            return len(self.test_data)
        if self.split is 'validate':
            return len(self.extra_data)  

    def _check_exists(self):
        return os.path.exists(os.path.join(os.path.join(self.root, self.file, (self.style + '_' + self.split + '.hdf5'))))
  

class USPS(data.Dataset):
    """USPS Dataset"""
    url = "https://www.kaggle.com/bistaumanga/usps-dataset"
    file =  'usps/usps.h5'

    def __init__(self, root, transform=None, train=True):
        super(USPS, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train

        if not self._check_exists():
            raise ValueError('Dataset file usps.h5 not found, download at: ' + self.url)

        with h5py.File(os.path.join(self.root, self.file), 'r') as f:
            if self.train is True:
                t = f.get('train')
                self.train_data = torch.tensor(t.get('data')[:])
                self.train_labels = torch.tensor(t.get('target')[:])  
            else:
                t = f.get('test')
                self.test_data = torch.tensor(t.get('data')[:])
                self.test_labels = torch.tensor(t.get('target')[:])  
        
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index].reshape(16,16), self.train_labels[index]
        else:
            img, target = self.test_data[index].reshape(16,16), self.test_labels[index]
        
        img = Image.fromarray(img.numpy())
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.train is True:
            return len(self.train_data)
        return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.file))


"""Dataset setting and data loader for MNIST-M.
Modified from
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
CREDIT: https://github.com/corenel
"""

class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'mnist_m'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self,
                 root, mnist_root="data",
                 transform=None, target_transform=None,
                 train=True,
                 download=True):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.training_file))
        else:
            self.test_data, self.test_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data) 

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
            os.path.exists(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print('Downloading ' + self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace('.gz', '')):
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), "rb") as f:
            mnist_m_data = pickle.load(f, encoding='bytes')
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root,
                                            train=True,
                                            download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root,
                                           train=False,
                                           download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


class mnist_dataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.image_data = data
        self.label = labels
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img = self.image_data[idx]
        target = self.label[idx]

        
        return img, target


def generate_degset(size=100, img_size=24):
    
    angles = [0, 15, 30, 45, 60, 75]

    data_path = './data'

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])

    tv = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])


    mnist_degs = []

    _train = torchvision.datasets.MNIST(
        data_path, train=True, download=True, transform=None)

    with torch.no_grad():
        for angle in angles:

            mnist_data = []
            mnist_label = []
            for i in range(0, 10):
                mnist_data.append(_train.train_data[_train.train_labels == i][np.random.choice(5000, size, replace=False)].numpy())
                mnist_label.append((_train.train_labels[:size] * 0) + i)

            mnist_data = np.vstack(mnist_data)
            mnist_label = torch.cat(mnist_label)

            tmp2 = []

            for i in range(0, mnist_data.shape[0]):
                tmp = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(img_size)])

                img = Image.fromarray(mnist_data[i], mode='L')
                img = tmp(img)
                img = torchvision.transforms.functional.rotate(img,  angle)
                img = tv(img)
                tmp2.append(img)

            mnist_degs.append((torch.stack(tmp2), mnist_label))
    return mnist_degs

def mnist_loader(deg, size, mnist_degs):
    data = []
    labels = []
    for i in range(0, 6):
        if i is not deg:
            data.append(mnist_degs[i][0])
            labels.append(mnist_degs[i][1])
    train = mnist_dataset(utils.to_3channels(torch.cat(data)), torch.cat(labels))
    test = mnist_dataset(utils.to_3channels(mnist_degs[deg][0]), mnist_degs[deg][1])
    return {
        'train': torch.utils.data.DataLoader(
            train, batch_size=size, shuffle=True,
            pin_memory=True, num_workers=2),
        'test' : torch.utils.data.DataLoader(
            test, batch_size=size, shuffle=False,
            num_workers=10
        )
    }

class FERGD(data.Dataset):
    url = "http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017"
    file = 'fergd/'
    styles =  ['art_painting', 'cartoon', 'photo', 'sketch']


    def __init__(self, root, transform=None, split='train', p=0.0):
        super(FERGD, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        
        r = 0
        
        if split is 'train':
            r = 5000
        elif split is 'validate':
            r = 3000
        else:
            r = 2000
            
        images = []
        labels = []
        
        for i in range(1, 8):
            for f in glob.iglob(os.path.join(self.root, self.file, str(p), split, str(i)+'/*')):
                images.append(np.asarray(Image.open(f).convert('RGB')))
                labels.append(i)
        
            
        if self.split is 'train':
            self.train_data = np.array(images)
            self.train_labels = np.array(labels)
        if self.split is 'valid':
            self.valid_data = np.array(images)
            self.valid_labels = np.array(labels)        
        if self.split is 'test':
            self.test_data = np.array(images)
            self.test_labels = np.array(labels)
                                             

    def __getitem__(self, index):
        if self.split is 'train':
            img, target = self.train_data[index], self.train_labels[index]
        if self.split is 'test':
            img, target = self.test_data[index], self.test_labels[index]
        if self.split is 'valid':
            img, target = self.valid_data[index], self.valid_labels[index]

        img = Image.fromarray(np.int8(img), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.split is 'train':
            return len(self.train_data)
        if self.split is 'test':
            return len(self.test_data)
        if self.split is 'valid':
            return len(self.valid_data)  

    def _check_exists(self):
        return os.path.exists(os.path.join(os.path.join(self.root, self.file, (self.style + '_' + self.split + '.hdf5'))))
  
