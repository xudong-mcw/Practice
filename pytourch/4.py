import torch
a = torch.rand(2,3)
b = torch.rand(3)
c = a+b
print(a)
print(b)
print(c)
print(c.shape)

a = torch.rand(2,1)
b = torch.rand(2)
c = a+b
print(a)
print(b)
print(c)
print(c.shape)

a = torch.rand(2,4,1,3)
b = torch.rand(4,2,3)
c = a+b
print(a)
print(b)
print(c)
print(c.shape)