import torch
dev = torch.device("cpu")
a = torch.tensor([2,2],device=dev,dtype=torch.float)
print(a)
print(a.type())
i = torch.tensor([[0,1,2,],[0,1,2]])
v = torch.tensor([1,2,3])
a = torch.sparse_coo_tensor(i,v,(4,4)).to_dense()
print(a)