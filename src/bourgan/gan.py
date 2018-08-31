#gan class
import torch
import torch.nn
from torch.utils.data import DataLoader

from bourgan.datasets.gaussianGridDatasetDataset import gaussianGridDataset
from bourgan.sampler.BourgainSampler import BourgainSampler


class BourGAN(object):
    def __init__(self):
        self.epoch = 2000
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
        

        self.show_step = 100

        #init network
        dim_outout = self.dataset.out_dim
        self.z_sampler = BourgainSampler(self.dataset.data)
        self.scale_factor, self.z_dim = self.z_sampler.scale, self.z_sampler.embedded_data.shape[1]

        self.G = DeepMLPG(self.z_dim, 128, self.dataset.out_dim)
        self.D = DeepMLPD(self.dataset.out_dim, 128, 1)


        self.G_opt = optim.Adam(self.G.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.D_opt = optim.Adam(self.D.parameters(), lr=1e-3, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()

        if self.use_gpu:
            print('using gpu!!')
            self.G.cuda()
            self.D.cuda()
            self.criterion = self.criterion.cuda()
            self.criterion_mse = self.criterion_mse.cuda()
            
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
                d_real_data = Variable(real_samples.float())
                real_samples = real_samples.numpy()
                if self.use_gpu:
                    d_real_data = d_real_data.cuda()
                d_real_decision = self.D(d_real_data)
                labels = Variable(torch.ones(d_real_decision.shape))
                if self.use_gpu:
                    labels = labels.cuda()
                
                d_real_loss = self.criterion(d_real_decision, labels)

                # Train D on fake
                latent_samples = self.z_sampler.sampling(self.batch_size)
                d_gen_input = Variable(latent_samples)
                
                if self.use_gpu:
                    d_gen_input = d_gen_input.cuda()
                d_fake_data = self.G(d_gen_input)

                d_fake_decision = self.D(d_fake_data)
                labels = Variable(torch.zeros(d_fake_decision.shape))
                if self.use_gpu:
                    labels = labels.cuda()
                d_fake_loss = self.criterion(d_fake_decision, labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_loss_np = d_loss.data.cpu().numpy() if self.use_gpu else d_loss.data.numpy()
                self.train_hist['D_loss'].append(d_loss_np[0])
                self.D_opt.step()

            
            for g_index in range(self.g_step):
                # Train G
                self.G.zero_grad()

                latent_samples = self.z_sampler.sampling(self.batch_size)

                g_gen_input = Variable(latent_samples)
                if self.use_gpu:
                    g_gen_input = g_gen_input.cuda()
                g_fake_data = self.G(g_gen_input)
                g_fake_decision = self.D(g_fake_data)
                labels = Variable(torch.ones(g_fake_decision.shape))
                if self.use_gpu:
                    labels = labels.cuda()

                gan_loss = self.criterion(g_fake_decision, labels)
                gan_loss_np = gan_loss.data.cpu().numpy() if self.use_gpu else gan_loss.data.numpy()
                self.train_hist['G_loss'].append(gan_loss_np[0])
              
                #add dist loss here
                latent_samples2 = self.z_sampler.sampling(self.batch_size)

                g_gen_input2 = Variable(latent_samples2)
                if self.use_gpu:
                    g_gen_input2 = g_gen_input2.cuda()
                z_dist = self.zdist(g_gen_input, g_gen_input2)
                g_fake_data2 = self.G(g_gen_input2)
                g_dist = self.gdist(g_fake_data, g_fake_data2)

                self.g_dists.append(g_dist.data.cpu().numpy())
                self.z_dists.append(z_dist.data.cpu().numpy())

       
                dist_loss = self.criterion_mse(torch.log(g_dist), torch.log(self.scale_factor*z_dist))
                dist_loss_np = dist_loss.data.cpu().numpy() if self.use_gpu else dist_loss.data.numpy()
                self.train_hist['dist_loss'].append(dist_loss_np[0])

                g_loss = gan_loss + self.alpha*dist_loss
            


                g_loss.backward()
                self.G_opt.step()

            if ep % self.show_step == 0:
                cuda = self.use_gpu
                loss_d_real = d_real_loss.data.cpu().numpy() if cuda else d_real_loss.data.numpy()
                loss_d_fake = d_fake_loss.data.cpu().numpy() if cuda else d_fake_loss.data.numpy()
                loss_g = g_loss.data.cpu().numpy() if cuda else g_loss.data.numpy()

                msg = 'Iteration {}: D_loss(real/fake): {}/{} G_loss: {}'.format(ep, loss_d_real[0], loss_d_fake[0], loss_g[0])
                print(msg)

