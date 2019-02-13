import numpy as np

import torch
from matplotlib import pyplot
from lib.plotter import plot_history
import os
import copy

def to_3channels(img):
    """To 3 channels

    Parameters:
        img : matrix img n x 1 x w x h
    
    Changes one channeled batch of images to three channeled in shape
        n x 3 x w x h
    """
    return img.repeat(1, 3, 1, 1)


def to_grayscale(batch):
    """To grayscale

    Parameters:
        batch : array of images [n x 3 x w x h] or [n x 1 x w x h]


    Changes images to grayscale if possible, returns array of images [n x 1 x w x h]
    """

    if batch.shape[1] == 1:
        return batch

    batch[:,0,:,:] *= 0.2126
    batch[:,1,:,:] *= 0.7152
    batch[:,2,:,:] *= 0.0722
    return (batch.sum(dim=1)[:,None,:])

def pac_label_to_string(id):
    """Pac label to string

    Parameters:
        id : int 
        label of picture from PACS database, ranges from 1-7

    Returns label mapped to string equivalent
    """
    m = ['dog', 'elephant', 'giraffe', 'gituar', 'horse', 'house', 'person']
    if (id < 1) or id > 7:
        raise ValueError('id not in 1-7 range')
    return m[id-1]    

def compute_error_rate(model, data_loader, cuda=True):
    model.eval()
    num_errs = 0.0
    num_examples = 0
    for x, y in data_loader:
        y = y.type(torch.LongTensor)
        if cuda:
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            outputs = model.forward(x)
            _, predictions = outputs.max(dim=1)
            num_errs += (predictions != y).sum().item()
            num_examples += x.size(0)
    return 100.0 * num_errs / num_examples

def train(model, data_loaders, optimizer, criterion, num_epochs=1,
          log_every=100, cuda=True, test_dataset='test'):
    if cuda:
        model.cuda()  
    
    iter_ = 0
    epoch = 0
    best_params = None
    best_val_err = np.inf
    history = {'train_losses': [], 'train_errs': [], 'val_errs': []}
    print('Training the model!')

    print("Params to learn:")
    
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)      

    print('You can interrupt it at any time.')
    try:
        while epoch < num_epochs:
            model.train()

            epoch += 1
            for x, y in data_loaders['train']:
                
                y = y.type(torch.LongTensor)
                
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                iter_ += 1

                optimizer.zero_grad()
                out = model.forward(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                _, predictions = out.max(dim=1)
                err_rate = 100.0 * (predictions != y).sum() / out.size(0)

                history['train_losses'].append(loss.item())
                history['train_errs'].append(err_rate.item())

                if iter_ % log_every == 0:
                    print("Minibatch {0: >6}  | loss {1: >5.2f} | err rate {2: >5.2f}%" \
                          .format(iter_, loss.item(), err_rate))

            
            model.eval()
            val_err_rate = compute_error_rate(model, data_loaders[test_dataset], cuda)
            history['val_errs'].append((iter_, val_err_rate))
            model.train()

            if val_err_rate < best_val_err:
                best_epoch = epoch
                best_val_err = val_err_rate
                best_params = copy.deepcopy(model.state_dict())
            m = "After epoch {0: >2} | valid err rate: {1: >5.2f}% | doing {2: >3} epochs" \
                .format(epoch, val_err_rate, num_epochs)
            print('{0}\n{1}\n{0}'.format('-' * len(m), m))
    except KeyboardInterrupt:
        pass
    if best_params is not None:
        print("\nLoading best params on validation set (epoch %d)\n" %(best_epoch))
        model.load_state_dict(best_params)
    plot_history(history)
    
def alex_classifier(num_classes=1000):
    return torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(256 * 6 * 6, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(4096, num_classes),
        torch.nn.LogSoftmax(dim=1)
    )
def save_model(model, name):
    root = os.path.expanduser('./models')
    torch.save(model.state_dict(), os.path.join(root, name))

def load_model(model, name):
    root = os.path.expanduser('./models')
    model.load_state_dict(torch.load(os.path.join(root, name)))    
    model.eval()

def list_models():
    root = os.path.expanduser('./models')    
    return os.listdir(root)
            
