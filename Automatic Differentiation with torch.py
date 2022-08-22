import torch
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3,requires_grad=True)
z = torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x,w) +b
print (z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x,w) + b

print(z.requires_grad)

z = torch.matmul(x,w)+b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(5,requires_grad=True)
print(inp)
print(inp+1)
out = (inp+1).pow(2)
print(out)
out.backward(torch.ones_like(inp),retain_graph=True)
print(f"First call \n{inp.grad}")
out.backward(torch.ones_like(inp),retain_graph=True)
print(f"\nSecond call \n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp),retain_graph=True)
print(f"\n Call after zeroing gradints\n{inp.grad}")