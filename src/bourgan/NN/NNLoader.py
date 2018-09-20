from bourgan.NN.MLP import *
import torch
import torch.nn
from torch import optim


def loadNN(nn_config, input_size, output_size):
    nn_name = nn_config['name']
    if nn_name == "DeepMLP_G":
        return DeepMLP_G(input_size, nn_config['hidden_size'], output_size)
    elif nn_name == "DeepMLP_D":
        return DeepMLP_D(input_size, nn_config['hidden_size'], output_size)
    else:
        raise ValueError("no NN called "+nn_config['name'])



def loadOpt(parameters, opt_config):
    opt_name = opt_config['name']
    if opt_name == 'adam':
        if opt_config['default'] == True:
            return optim.Adam(parameters, lr=1e-3, betas=(0.5, 0.999))
        else:
            return optim.Adam(parameters, lr=opt_config['lr'], betas=opt_config['betas'])
    else:
        raise ValueError("no opt called "+opt_name)