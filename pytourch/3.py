
import torch
a = torch.rand(2,3)
b = torch.rand(2,3)

print(a)
print(b)

print(a+b)
print(torch.add(a,b))
print(a.add(b))
print(a)
print(a.add_(b))
print(a)

print("=======")

print(a-b)
print(torch.sub(a,b))
print(a.sub(b))
print(a)
print(a.sub_(b))
print(a)

print("=======")
print(a*b)
print(a.mul(b))
print(torch.mul(a,b))
print(a)
print(a.mul_(b))
print(a)

print("========")
print(a/b)
print(torch.div(a,b))
print(a.div(b))
print(a)
print(a.div_(b))
print(a)

a = torch.ones(2,1)
b = torch.ones(1,2)

print(a@b)
print(torch.matmul(a,b))
print(a.mm(b))
print(torch.mm(a,b))
print(a.matmul(b))

a = torch.ones(1,2,3,4)
b= torch.ones(1,2,4,3)

print(a.matmul(b).shape)

a = torch.tensor([1,2])
print(torch.pow(a,3))
print(a.pow(3))
print(a**3)
print(a.pow_(3))

a = torch.tensor([1,2])
print(torch.exp(a))
print(a.exp())

a = torch.tensor([10,2],dtype=torch.float)
print(torch.log(a))

print(torch.sqrt(a))
print(a.sqrt())