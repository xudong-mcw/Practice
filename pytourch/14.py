import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn,optim
#from lenet5 import Lenet5
from resnet import ResNet18

def main():
    batchsz =32
    cifar_train= datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ]),download=True)
    cifar_train = DataLoader(cifar_train,batch_size=batchsz,shuffle= True)

    cifar_test= datasets.CIFAR10('cifar',False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ]),download=True)
    cifar_test = DataLoader(cifar_train,batch_size=batchsz,shuffle= True)

    x, label = iter(cifar_train).next()
    print('x:',x.shape,'lable:',label.shape)

    device = torch.device('cuda')
    #model = Lenet5().to(device)
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    print(model)

    for epoch in range(1000):
        for (x,label) in enumerate(cifar_test):
            x,label = x.to(device),label.to(device)
            logits = model(x)
            loss = criteon(logits,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch,loss.item())

        model.eval()
        with torch.no_grad():
            total_cirrect = 0
            total_num = 0
            for x ,label in cifar_test:
                x,label = x.to(device),label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_cirrect += torch.eq(pred,label).float().sum().item()
                total_num += x.size(0)

            acc = total_cirrect/total_num
            print(epoch,acc)    

if __name__ == '__main__':
    main()