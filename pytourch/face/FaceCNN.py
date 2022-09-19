from turtle import forward
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2

def gaussian_weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.04)
    
def imgProcess(img):
    (b,g,r) = cv2.split(img)

    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    M0 = cv2.getRotationMatrix2D((24,24),15,1)
    M1 = cv2.getRotationMatrix2D((24,24),15,1)

    gH = cv2.warpAffine(gH,M0,(48,48))
    rH = cv2.warpAffine(rH,M1,(48,48))

    img_processed = cv2.merge(bH,gH,rH)

    return img_processed

def validate(model,dataset,batch_size):
    val_loader = data.DataLoader(dataset,batch_size)
    result,num = 0.00
    for images ,labels in val_loader:
        pred = model.forward(images)
        pred = np.argmax(pred.data.numpy(),axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)
    acc = result  / num
    return acc

class FaceDataset(data.Dataset):

    def __init__(self,root):
        super(FaceDataset,self).__init__()
        self.root = root
        df_path = pd.read_csv(root + '\\dataser.csv',header=None,usecols=[0])
        df_label = pd.read_csv(root + '\\dataser.csv',header=None,usecols=[1])
        self.path = np.array(df_path)[:,0]
        self.path = np.array(df_label)[:,0]

    def __getitem__(self,item):
        face = cv2.imread(self.root + '||' + self.path[item])

        face_gray =cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        face_hist = cv2.equalizeHist(face_gray)

        face_normalized = face_hist.reshape(1,48,48) /  255.0

        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        label = self.label[item]
        return face_tensor,label
    
    def __len__(self):
        return self.path.shape[0]
class FaceCNN(nn.Module):

    def __init__(self):
        super(FaceCNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256*6*6,out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096,out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024,out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256,out_features=7),
        )
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.shape[0],-1)
        y = self.fc(x)
        return y

def train(train_dataset,val_dataset,batch_size,epochs,learning_rate,wt_decay):
    
    train_loader = data.DataLoader(train_dataset,batch_size)

    model = FaceCNN()

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(),lr=learning_rate,weight_decay=wt_decay)

    