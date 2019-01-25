import numpy as np
import torch

def to_3channels(img):
    """To 3 channels

    Parameters:
        img : matrix img n x 1 x w x h
    
    Changes one channeled batch of images to three channeled in shape
        n x 3 x w x h
    """
    #b, _, w, h = img.shape
    #return torch.tensor(np.stack((img,)*3, axis=1)).reshape(b, 3, w, h)
    return img.repeat(1, 3, 1, 1)


def to_grayscale(batch):
    """To grayscale

    Parameters:
        batch : array of images [n x 3 x w x h]

    Changes images to grayscale, returns array of images [n x 1 x w x h]
    """
    batch[:,0,:,:] *= 0.2126
    batch[:,1,:,:] *= 0.7152
    batch[:,2,:,:] *= 0.0722
    return (batch.sum(dim=1)[:,None,:])