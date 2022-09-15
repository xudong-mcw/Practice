from cProfile import label
from email.policy import default
from xml.etree.ElementInclude import default_loader
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import numpy as np

label_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

label_dict = {}

for idx,name in enumerate(label_name):
    label_dict[name] = idx

def default_loader(path):
    return Image.open(path).convert("RGB")

train_transfor = transforms.Compose([
    transforms.RandomResizedCrop((28,28)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.RandomGrayscale(0.1),
    transforms.ColorJitter(0.3,0.3,0.3,0.3),
    transforms.ToTensor()
])
class MyDataSet(Dataset):
    def __init__(self,im_list,transform=None,loader= default_loader):
        super(MyDataSet,self).__init__()
        imgs = []

        for im_item in im_list:
            "/home/kuan/dataset/CIFAR10/TRAIN/"\
            "airplane/aeroplane_s_000021.png"
            im_label_name = im_item.split("/")[-2]
            imgs.append([im_item,label_dict[im_label_name]])
        
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
    
    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data,im_label

    def __len__(self):
        return len(self.imgs)
