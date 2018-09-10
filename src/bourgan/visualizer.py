import numpy as np
from matplotlib import pyplot as plt
import torch

from bourgan.utils import *


#visualizer
def plot_outputdata(out_data, newfig=False, dim1=0, dim2=1, marker='.', alpha=0.1):
    if newfig:
        fig = plt.figure()
    if isinstance(out_data, torch.Tensor):
        out = out_data.data.numpy()
    else:
        out = out_data
    plt.plot(out[:,dim1], out[:,dim2], marker, alpha=alpha)
    plt.axis('equal')

def visualize_G(nets, N, show_dataset=True, kde=False, dim1=0, dim2=1, alpha=0.1):
    samples = nets.z_sampler.sampling(N)
    
    in_data = samples.to(dtype=torch.float32, device=nets.device)

    out_data = nets.G(in_data)
    out_data = out_data.data.cpu().numpy()
    data = nets.dataset.data
    plot_outputdata(data, dim1=dim1, dim2=dim2, marker='kx', alpha=alpha)
    plot_outputdata(out_data, dim1=dim1, dim2=dim2, marker='.', alpha=alpha)
    plt.show()
    return data, out_data

def visualize_loss(nets):
    plt.figure()
    plt.subplot(311)
    if len(nets.train_hist['G_loss'])>0:
        plt.plot(nets.train_hist['G_loss'])
        plt.subplot(312)
    if len(nets.train_hist['D_loss'])>0:
        plt.plot(nets.train_hist['D_loss'])
        plt.subplot(313)
    if len(nets.train_hist['dist_loss'])>0:
        plt.plot(nets.train_hist['dist_loss'])
        plt.show()
        
def plot_sampler_simple(sampler, n, dim1=0, dim2=1):
    samples = sampler.sampling(n)
    samples = samples.numpy()
    plt.plot(samples[:, dim1], samples[:, dim2], 'o', alpha=0.1)
    plt.show()


def plot_sampler(sampler, n, tsne=False, kde=False, dim1=0, dim2=1, rotate=True, origin_data=False, alpha=0.1):
    samples = sampler.sampling(n)
    samples = samples.numpy()
    if origin_data:
        samples = sampler.embedded_data

    if rotate:
        ortho = random_orthonomal(samples.shape[1])
        samples = samples.dot(ortho)
    if tsne:
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=2).fit_transform(samples)
        plt.plot(X_embedded[:,0], X_embedded[:,1], 'o', alpha=alpha)
        plt.show()
    elif kde:
        plt.hist2d(samples[:, dim1], samples[:, dim2], (50, 50), cmap=plt.cm.jet)
    else:
        plt.plot(samples[:, dim1], samples[:, dim2], 'o', alpha=alpha)
        plt.show()