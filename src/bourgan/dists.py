import numpy
import scipy
from scipy.spatial.distance import pdist, squareform
import torch

def pairwise_dist(data, dist, p=None):
    if not isinstance(dist, str):
        raise ValueError('dist must be str')
    
    if dist == 'l2':
        return squareform(pdist(data, metric='euclidean'))
    elif dist == 'lp':
        return squareform(pdist(data, metric='minkowski', p=p))


def pairwise_dist_generic(data, distfunc):
    n = data.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j >= i:
                continue    

            dist[i, j] = distfunc(data[i, :], data[j, :])
            dist[j, i] = dist[i, j]
    
    return

    
def dist_l2(a, b):
    a = a.view((a.shape[0], -1))
    b = b.view((b.shape[0], -1))
    return torch.norm(a-b, p=2, dim=1)

def dist_l1(a, b):
    a = a.view((a.shape[0], -1))
    b = b.view((b.shape[0], -1))
    return torch.norm(a-b, p=1, dim=1)