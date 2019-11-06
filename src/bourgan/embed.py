import numpy as np
import scipy


def bourgain_embedding(data, p, m, distmat):
    """
    bourgain embedding main function.
    Args:
        data (ndarray): Input data for embedding. Shape must be nxm, 
                        where n is the number of data points, m is the data dimension.
        p, m (float): bourgain embedding hyperparameters.
        distmat (ndarray): Distance matrix for data, shape must be nxn.
    
    Returns:
        ans (ndarray): results for bourgain embedding, shape must be nxk, where k is the
            latent space dimension.

    """
    assert(p>0 and p<1)
    assert(isinstance(m, int))
    n = data.shape[0]
    K = np.ceil(np.log(n)/np.log(1/p))
    S={}
    for j in range(int(K)):
        for i in range(m):
            S[str(i)+str('_')+str(j)]=[]
            prob = np.power(p, j+1)
            rand_num = np.random.rand(n)
            good = rand_num<prob
            good = np.argwhere(good==True).reshape((-1))
            S[str(i)+str('_')+str(j)].append(good)


    ans = np.zeros((n, int(K)*m))

    for (c, x) in enumerate(data):
        fx = np.zeros((m, int(K)))
        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                fx[i, j] = mindist(c, S[str(i)+str('_')+str(j)], distmat)

        fx = fx.reshape(-1)
        ans[c, :] = fx

    ans = ans - np.mean(ans, axis=0)
    dists = np.linalg.norm(ans, ord='fro')/np.sqrt(ans.shape[0])
    ans = ans/dists * np.sqrt(ans.shape[1])
    return ans

def mindist(x_id, idxset, distmat):
    """
    helper function to find the minimal distance in a given point set
    Args:
        x_id (int): id for reference point
        idxset (list): ids for the point set to test
        distmat (ndarray): distance matrix for all points
    
    Returns:
        mindist (float): minimal distance
    """
    d = distmat[x_id, idxset[0]]
    if d.shape[0] == 0:
        return 0
    else:
        return np.min(d)
