"""
This function contains code to images and weight matrices.
imported from https://github.com/janchorowski/nn_assignments/blob/nn18/common/plotting.py
"""

import numpy as np
from matplotlib import pyplot
import pylab

import seaborn as sns
sns.set_style('whitegrid')

def scale_mat(mat, lower=0., upper=1.):
    """
    Scale all linearly all elements in a mtrix into a given range.
    """
    ret = mat - np.min(mat)
    return ret * ((upper-lower) / np.max(ret)) + lower

def get_grid(num_elem, prop=(9,16)):
    """
    Find grid proportions that would accomodate given number of elements.
    """
    cols = np.ceil(np.sqrt(1. * num_elem * prop[1] / prop[0]))
    rows = np.ceil(1. * num_elem / cols)
    while cols != np.ceil(1. * num_elem / rows):
        cols = np.ceil(1. * num_elem / rows)
        rows = np.ceil(1. * num_elem / cols)
    return int(rows), int(cols)

def plot_mat(mat, scaleIndividual=True, colorbar=False, prop=(9,16), gutters=2,
             scale_fun=scale_mat, **kwargs):
    """
    Plot an image for each entry in the tensor.
    Inputs
    ------
    mat: 4D tensor, n_images x n_channels x rows x columns
    """
    nSamples, nChannels, r, c = mat.shape
    gr, gc =  get_grid(nSamples, (prop[0]*c, prop[1]*r))
    toPlot = np.zeros((int(gr*r+(gr-1)*gutters), int(gc*c + (gc-1)*gutters), nChannels) ) + np.NaN
    for s in range(nSamples):
        pr = s // gc
        pc = s - (pr*gc)
        small_img = mat[s,:,:,:].transpose(1,2,0)
        if scaleIndividual:
            small_img = scale_fun(small_img)
        toPlot[int(pr*(r+gutters)):int(pr*(r+gutters)+r),
               int(pc*(c+gutters)):int(pc*(c+gutters)+c),:] = small_img
    if nChannels==1:
        pyplot.imshow(toPlot[:,:,0], interpolation='nearest', **kwargs)
    else:
        pyplot.imshow(toPlot, interpolation='nearest', **kwargs)
    if colorbar:
        pyplot.colorbar()
    pyplot.axis('off')



def plot_history(history):
    pyplot.figure(figsize=(16, 4))
    pyplot.subplot(1,2,1)
    train_loss = np.array(history['train_losses'])
    pyplot.semilogy(np.arange(train_loss.shape[0]), train_loss, label='batch train loss')
    pyplot.legend()
        
    pyplot.subplot(1,2,2)
    train_errs = np.array(history['train_errs'])
    pyplot.plot(np.arange(train_errs.shape[0]), train_errs, label='batch train error rate')
    val_errs = np.array(history['val_errs'])
    pyplot.plot(val_errs[:,0], val_errs[:,1], label='validation error rate', color='r')
    pyplot.ylim(0,20)
    pyplot.legend()    