import torch
import numpy as np
import re
ff = open('D:\pytorch\Practice\pytourch\housing.data').readlines()
data = []
for item in ff:
    out = re.sub(r"\s{2,}"," ",item).strip()
    print(out)
    data.append(out.split(" "))

data = np.array(data).astype(np.float32)
print(data)

y = data[:,-1]
x = data[:,0:-1]

X_train = x[0:496,...]
Y_train = y[0:496,...]

X_test = x[496:,...]
Y_test = y[496:,...]
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_output):
        super(Net,self).__init__()
        self.predict= torch.nn.Linear(n_feature,n_output)

    def forward(self,x):
        out = self.predict(x)
        return out

net= Net(13,1)

loss_func = torch.nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(),lr=0.0001)

for i in range(1000):
    x_data = torch.tensor(X_train,dtype=torch.float32)
    y_data = torch.tensor(Y_train,dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred,y_data) * 0.001
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("item:{},loss:{}".format(i,loss))
    print(pred[0:10])
    print(y_data[0:10])