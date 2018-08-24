import numpy as np
from matplotlib import pyplot as plt

#visualizer
def plot_outputdata(out_data, newfig=False, dim1=0, dim2=1,marker='.'):
    if newfig:
        fig = plt.figure()
    if isinstance(out_data, torch.Tensor):
        out = out_data.data.numpy()
    else:
        out = out_data
    plt.plot(out[:,dim1], out[:,dim2], marker)
    plt.axis('equal')

def visualize_G(nets, N, show_dataset=True, kde=False, dim1=0, dim2=1):
    uni = nets.z_sampler.sampling(N)
    if isinstance(uni, np.ndarray):
        uni = torch.from_numpy(uni.float())

    if nets.use_gpu:
        in_data = Variable(uni).cuda()
    else:
        in_data = Variable(uni)

    out_data = nets.G(in_data)
    out_data = out_data.data.cpu().numpy()
    data = nets.dataset.data
    plot_outputdata(data, dim1=dim1, dim2=dim2, marker='kx')
    plot_outputdata(out_data, dim1=dim1, dim2=dim2, marker='.')
    plt.show()
    return data, out_data

def visualize_loss(nets):
    plt.figure()
    plt.subplot(311)
    if len(nets.train_hist['G_loss'])>0:
        plt.plot(nets.train_hist['G_loss'])
        plt.subplot(312)
    if len(nets.train_hist['D_loss'])>0:
        plt.plot(nets.train_hist['D_loss'])
        plt.subplot(313)
    if len(nets.train_hist['dist_loss'])>0:
        plt.plot(nets.train_hist['dist_loss'])
        plt.show()
        
def plot_sampler(sampler, n, dim1=0, dim2=1):
    samples = sampler.sampling(n)
    samples = samples.numpy()
    plt.plot(samples[:, dim1], samples[:, dim2], 'o', alpha=0.1)
    plt.show()