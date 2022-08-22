import torch
a = torch.tensor([[1,2],[3,4]])
print(a)
print(a.type())

a = torch.zeros(2,3)
print(a)
print(a.type())

a = torch.ones(2,3)
print(a)
print(a.type())

b = torch.Tensor(5,5)
print(b)
print(b.type())

b = torch.zeros_like(b)
print(b)
print(b.type())


b= torch.eye(2,3)
print(b)
print(b.type())

a = torch.rand(2,2)
print(a)
print(a.type())

a = torch.normal(mean=torch.rand(5),std=torch.rand(5))
print(a)
print(a.type())

a = torch.Tensor(2,2).uniform_(-1,1)
print(a)
print(a.type())

a = torch.arange(0,11,2)
print(a)
print(a.type())

a = torch.linspace(2,10,4)
print(a)
print(a.type())

a = torch.randperm(10)
print(a)
print(a.type())

import numpy as np
