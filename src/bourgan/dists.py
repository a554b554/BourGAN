import numpy
import scipy

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

    