import torch

x = torch.DoubleTensor([1,1,1])
print(x.type())
print(torch.cuda.device_count())


