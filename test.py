import torch
import numpy as np

b = torch.tensor([[1, 2]])
print(b.size())
c = torch.stack([b, b], dim=0)
print(c.shape)

d = torch.cat([b, b], dim=0)
print(d.shape)

e = torch.stack([b], dim=0)
print(e.shape)