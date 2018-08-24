import numpy
import scipy
from scipy.spatial.distance import pdist, squareform


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

    