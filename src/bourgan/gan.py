#gan class
import torch
import torch.nn
from torch.utils.data import DataLoader
from torch import optim

from bourgan.datasets.gaussianGridDatasetDataset import gaussianGridDataset
from bourgan.sampler.BourgainSampler import BourgainSampler
from bourgan.NN.MLP import *
from bourgan.dists import *
from bourgan.visualizer import *

class BourGAN(object):
    def __init__(self):
        self.epoch = 5000
        self.batch_size = 128
        self.alpha = 0.1

    
        self.dataset = gaussianGridDataset(5, 50, 0.01)
        
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.use_gpu =True

        #frequency for alternating gan
        self.g_step = 1
        self.d_step = 1

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['dist_loss'] = []

        self.g_dists=[]
        self.z_dists=[]

        self.device = torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.show_step = 100


        #init network
        dim_outout = self.dataset.out_dim
        self.z_sampler = BourgainSampler(self.dataset.data)
        self.scale_factor, self.z_dim = self.z_sampler.scale, self.z_sampler.embedded_data.shape[1]


        self.G = DeepMLP_G(self.z_dim, 128, self.dataset.out_dim)
        self.D = DeepMLP_D(self.dataset.out_dim, 128, 1)

        self.G_opt = optim.Adam(self.G.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.D_opt = optim.Adam(self.D.parameters(), lr=1e-3, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()

        #convert device
        self.G.to(self.device)
        self.D.to(self.device)
        # self.criterion
        self.zdist = dist_l2 
        self.gdist = dist_l2


   

    def train(self, iters=0):
        print('training start')
        pbar = range(self.epoch)
        
        if iters != 0:
            pbar = range(iters)
        for ep in pbar:
            if ep % self.show_step == 0:
                real_data, fake_data = visualize_G(self, 5000, dim1=0, dim2=1)
                eppp = ep // self.show_step

            for d_index in range(self.d_step):
                # Train D

                self.D.zero_grad()

                # Train D on real
                real_samples = next(iter(self.dataloader))
                if isinstance(real_samples, list):
                    real_samples = real_samples[0]
                
                d_real_data = real_samples.to(dtype=torch.float32, device=self.device)

                d_real_decision = self.D(d_real_data)
                labels = torch.ones(d_real_decision.shape, dtype=torch.float32, device=self.device)
                
                d_real_loss = self.criterion(d_real_decision, labels)

                # Train D on fake
                latent_samples = self.z_sampler.sampling(self.batch_size)
                d_gen_input = latent_samples.to(dtype=torch.float32, device=self.device)
                d_fake_data = self.G(d_gen_input)
                d_fake_decision = self.D(d_fake_data)
                labels = torch.zeros(d_fake_decision.shape, dtype=torch.float32, device=self.device)
     
                d_fake_loss = self.criterion(d_fake_decision, labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_loss_np = d_loss.item()
                self.train_hist['D_loss'].append(d_loss_np)
                self.D_opt.step()

            
            for g_index in range(self.g_step):
                # Train G
                self.G.zero_grad()

                latent_samples = self.z_sampler.sampling(self.batch_size)

                g_gen_input = latent_samples.to(dtype=torch.float32, device=self.device)
                g_fake_data = self.G(g_gen_input)
                g_fake_decision = self.D(g_fake_data)
                labels = torch.ones(g_fake_decision.shape, dtype=torch.float32, device=self.device)
               
                gan_loss = self.criterion(g_fake_decision, labels)
                gan_loss_np = gan_loss.item()
                # gan_loss_np = gan_loss.data.cpu().numpy() if self.use_gpu else gan_loss.data.numpy()
                self.train_hist['G_loss'].append(gan_loss_np)
              
                #add dist loss here
                latent_samples2 = self.z_sampler.sampling(self.batch_size)

                g_gen_input2 = latent_samples2.to(dtype=torch.float32, device=self.device)

                z_dist = self.zdist(g_gen_input, g_gen_input2)
                g_fake_data2 = self.G(g_gen_input2)
                g_dist = self.gdist(g_fake_data, g_fake_data2)

                self.g_dists.append(g_dist.to("cpu").detach().numpy())
                self.z_dists.append(z_dist.to("cpu").detach().numpy())

                dist_loss = self.criterion_mse(torch.log(g_dist), torch.log(self.scale_factor*z_dist))
                dist_loss_np = dist_loss.item()
                self.train_hist['dist_loss'].append(dist_loss_np)
                g_loss = gan_loss + self.alpha*dist_loss
                g_loss.backward()
                self.G_opt.step()

            if ep % self.show_step == 0:
                cuda = self.use_gpu
                loss_d_real = d_real_loss.item()
                loss_d_fake = d_fake_loss.item()
                loss_g = g_loss.item()

                msg = 'Iteration {}: D_loss(real/fake): {}/{} G_loss: {}'.format(ep, loss_d_real, loss_d_fake, loss_g)
                print(msg)

