
import pickle

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label_name = ["airplane",
                "automobible",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck"]

import glob
import numpy as np
train_list = glob.glob("D:\pytorch\cifar-10-python\cifar-10-batches-py\data_batch_*")

print(train_list)

for l in train_list:
    print(l)
    l_data = unpickle(l)
    print(l_data)
    print(l_data.keys())

    for im_dix,im_data in enumerate(l_data[b'data']):
        print(im_dix)
        print(im_data)

        im_label = l_data[b'data'][im_dix]
        im_name = l_data[b'filenames'][im_dix]

        print(im_label,im_name,im_data)

        im_lable_name = label_name[im_label]
        im_data = np.reshape(im_data,[3,32,32])
        im_data = np.transpose(im_data,[1,2,0])