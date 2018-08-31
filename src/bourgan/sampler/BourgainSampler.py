from bourgan.embed import *
from bourgan.dists import *


	# if compute_alpha:
	# 	l1 = pair_wise_l2(ans)
	# 	for i in range(l1.shape[0]):
	# 		l1[i,i] = 1
	# 	div1 = np.sqrt(np.divide(distances, l1))
	# 	return ans, np.max(div1)

class BourgainSampler(object):
    def __init__(self, data, dist='l2'):
        self.name = "bourgain"
        p = 0.5
        m = 5
        self.eps = 0.1
        self.origin_data = data
        distmat = pairwise_dist(data, dist=dist)
        self.embedded_data = bourgain_embedding(data, p=p, m=m, distmat=distmat)
        self.scale = float(self.get_scale(self.embedded_data, distmat))
        print('scale factor:', scale_factor)

        
    def sampling(self, n):   #bourgain sampling
        num_data = self.embedded_data.shape[0]
        sampled_idx = np.random.choice(num_data, n)
        sampled_data = self.embedded_data[sampled_idx, :]
        noise = np.random.normal(scale=self.eps, size=sampled_data.shape)
        sampled_data = sampled_data + noise
        sampled_data = torch.from_numpy(sampled_data).float()
        return sampled_data

    def get_scale(self, embedded_data, distmat):
        l2 = pairwise_dist(embedded_data, 'l2')
        for i in range(l2.shape[0]):
            l2[i, i] = 1
        div1 = np.sqrt(np.divide(distmat, l2))
        return np.max(div1)

    # def compute_bourgan_scale()