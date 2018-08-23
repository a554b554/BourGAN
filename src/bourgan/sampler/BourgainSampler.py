from bourgan.embed import *


class BourgainSampler(object):
    def __init__(self, nets, path=None):
        self.name = "bourgain"
        p = 0.5
        m = 5
        self.eps = 0.1
        self.embedded_data, self.scale = bourgain_embedding(nets.dataset.data, p=p, m=m)
        self.origin_data = nets.dataset.data
        nets.scale_factor = float(self.scale)
        print('scale factor:', nets.scale_factor)
        nets.z_dim = self.embedded_data.shape[1]
        
    def sampling(self, n):   #bourgain sampling
        num_data = self.embedded_data.shape[0]
        sampled_idx = np.random.choice(num_data, n)
        sampled_data = self.embedded_data[sampled_idx, :]
        noise = np.random.normal(scale=self.eps, size=sampled_data.shape)
        sampled_data = sampled_data + noise
        sampled_data = torch.from_numpy(sampled_data).float()
        return sampled_data