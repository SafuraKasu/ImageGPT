import torch

a = torch.tensor([1,2])
b= torch.tensor([3,4])

c = torch.stack((a,b))
print(c)

print(torch.stack([a,b]))
print(torch.cat([a,b]))