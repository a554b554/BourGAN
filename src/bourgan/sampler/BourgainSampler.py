from bourgan.embed import *
from bourgan.dists import *
import torch
import numpy as np

class BourgainSampler(object):
    def __init__(self, data, path=None, dist='l2'):
        if path is not None:
            self.load(path)
            return

        self.name = "bourgain"
        p = 0.5
        m = 5
        self.eps = 0.01
        self.origin_data = data
        distmat = pairwise_dist(data, dist=dist)
        self.embedded_data = bourgain_embedding(data, p=p, m=m, distmat=distmat)
        #the data stored in Sampler are numpy array
        self.scale = float(self.get_scale(self.embedded_data, distmat))
        print('scale factor:', self.scale)

        
    def sampling(self, n):   #bourgain sampling
        num_data = self.embedded_data.shape[0]
        sampled_idx = np.random.choice(num_data, n)
        sampled_data = self.embedded_data[sampled_idx, :]
        noise = np.random.normal(scale=self.eps, size=sampled_data.shape)
        sampled_data = sampled_data + noise
        sampled_data = torch.from_numpy(sampled_data).float()
        #return torch.Tensor
        return sampled_data


    def get_scale(self, embedded_data, distmat):
        l2 = pairwise_dist(embedded_data, 'l2')
        for i in range(l2.shape[0]):
            l2[i, i] = 1
        div1 = np.sqrt(np.divide(distmat, l2))
        return np.max(div1)

    def save(self, path):
        np.savez(path, eps=self.eps, embed=self.embedded_data, scale=self.scale, origin_data=self.origin_data)

    def load(self, path):
        ff = np.load(path)
        self.embedded_data = ff['embed']
        self.eps = ff['eps']
        self.scale = ff['scale']
        self.origin_data = ff['origin_data']
        self.name = "bourgain"


class UniformSampler(object):
    def __init__(self, dim):
        self.dim = dim
        self.name = "uniform"
    
    def sampling(self, n):
        return torch.rand(n, self.dim)

class GaussianSampler(object):
    def __init__(self, dim):
        self.dim = dim
        self.name = "gaussian"
    
    def sampling(self, n):
        return torch.randn(n, self.dim)


def loadSampler(sampler_config, dataset=None):
    sampler_name = sampler_config['name']
    if sampler_name == "uniform":
        return UniformSampler(sampler_config['dim'])
    elif sampler_name == "gaussian":
        return GaussianSampler(sampler_config['dim'])
    elif sampler_name == "bourgain":
        return BourgainSampler(data=dataset.data, path=sampler_config['path'], dist=sampler_config['dist'])
    else:
        raise ValueError("no such sampler called "+sampler_name)