from hashlib import sha1
import torch
print("torch.cat")
a = torch.zeros((2,4))
b = torch.ones((2,4))
out = torch.cat((a,b),dim=0)
print(a)
print(b)
print(out)
print("torch.stack")
a = torch.linspace(1,6,6).view(2,3)
b = torch.linspace(7,12,6).view(2,3)
print(a,b)
out = torch.stack((a,b),dim=0)
print(out)
print(out.shape)

a = torch.rand((3,4))
out = torch.chunk(a,2,dim=0)

print(out[0],out[0].shape)
print(out[1],out[1].shape)

out = torch.split(a,2,dim=1)
print(out)

a = torch.full((2,3),10)
print(a)