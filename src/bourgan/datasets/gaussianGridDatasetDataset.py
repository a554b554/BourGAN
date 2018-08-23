import numpy as np
from torch.utils.data import Dataset, DataLoader

class gaussianGridDataset(Dataset):
    def __init__(self, n, n_data, sig):
        self.grid = np.linspace(-4, 4, n)
        self.data = None
        for i in range(n):
            mean_x = self.grid[i]
            for j in range(n):
                mean_y = self.grid[j]
                if self.data is None:
                    self.data = np.random.multivariate_normal((mean_x, mean_y), cov=[[sig, 0.0], [0.0, sig]], size=n_data)
                else:
                    self.data = np.concatenate((self.data, np.random.multivariate_normal((mean_x, mean_y), cov=[[sig, 0.0], [0.0, sig]], size=n_data)), axis=0)
        self.out_dim = 2
        self.n_data = self.data.shape[0]

    
    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data