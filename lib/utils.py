import numpy as np
import torch

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
    m = ['dog', 'elephant', 'giraffe', 'gituar', 'horse', 'house','person']
    if (id < 1) or id > 7:
        raise ValueError('id not in 1-7 range')
    return m[id-1]    