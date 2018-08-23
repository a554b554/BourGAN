import sys
sys.path.append('../')
from bourgan.sampler import BourgainSampler
import numpy as np



if __name__ == '__main__':
    c = BourgainSampler.BourgainSampler(3)
    print('x')