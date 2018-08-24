from bourgan.embed import *
from bourgan.dists import *


class BourgainSampler(object):
    def __init__(self, data, dist='l2'):
        self.name = "bourgain"
        p = 0.5
        m = 5
        self.eps = 0.1
        self.origin_data = data
        distmat = pairwise_dist(data, dist=dist)
        self.embedded_data = bourgain_embedding(data, p=p, m=m, distmat=distmat)
        nets.scale_factor = float(self.scale)
        print('scale factor:', nets.scale_factor)
        nets.z_dim = self.embedded_data.shape[1]
        return scale_factor, z_dim
        
    def sampling(self, n):   #bourgain sampling
        num_data = self.embedded_data.shape[0]
        sampled_idx = np.random.choice(num_data, n)
        sampled_data = self.embedded_data[sampled_idx, :]
        noise = np.random.normal(scale=self.eps, size=sampled_data.shape)
        sampled_data = sampled_data + noise
        sampled_data = torch.from_numpy(sampled_data).float()
        return sampled_data

    # def compute_bourgan_scale()