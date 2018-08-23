import sys
sys.path.append('../')
from bourgan.embed import *
import numpy as np
from scipy.spatial.distance import pdist, squareform


def bourgain_embedding_test():
    data = np.random.randn(1000, 200)
    dmat = squareform(pdist(data, metric='euclidean'))
    print(dmat.shape)
    embedded = bourgain_embedding(data, p=0.5, m=5, distmat=dmat)
    print(embedded.shape)


if __name__ == '__main__':
    bourgain_embedding_test()