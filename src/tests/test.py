import sys
sys.path.append('../')
from bourgan.sampler.BourgainSampler import BourgainSampler
import numpy as np



if __name__ == '__main__':
    c = BourgainSampler(3)
    print('x')