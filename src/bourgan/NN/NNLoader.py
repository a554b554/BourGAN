from bourgan.NN.MLP import *



def loadNN(nn_config):
    nn_name = nn_config['name']
    if nn_name == "DeepMLP_G":
        return DeepMLP_G(nn_config['input_size'], nn_config['hidden_size'], nn_config['output_size'])
    elif nn_name == "DeepMLP_G":
        return DeepMLP_D(nn_config['input_size'], nn_config['hidden_size'], nn_config['output_size'])
    else:
        raise ValueError("no NN called "+nn_config['name'])