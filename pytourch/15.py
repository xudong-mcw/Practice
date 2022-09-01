from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(2,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

        tmp = torch.randn(2,3,32,32)
        out = self.conv_unit(tmp)
        print('conv out:',out.shape)

        self.criteon = nn.CrossEntropyLoss()

        
    def forward(self,x):
        batchsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz,16*5*5)
        #logits = self.fc_unit(x)
        #pred = F.softmax(logits,dim=1)

def main():
    net = Lenet5()
    tmp = torch.randn(2,3,32,32)
    out = net(tmp)
    