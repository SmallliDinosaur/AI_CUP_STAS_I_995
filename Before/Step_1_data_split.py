# data split

import numpy as np
import shutil
import os

from_img_dir = './OBJ_Train_Datasets/Train_Images/'
from_xml_dir = './OBJ_Train_Datasets/Train_Annotations/'
img_list = os.listdir(from_img_dir)
print('Files include ',img_list[:4])
n_img = len(img_list)
print('Number of Images =', n_img)

np.random.shuffle(img_list)
print('After shuffling, files became', img_list[:4])

# data split for training and testing set
split_ratio = 0.2 # Train:Val = 8:2
n_train = round(n_img*(1-split_ratio))
n_val = n_img - n_train

train_list = img_list[:n_train]
val_list = img_list[n_train:]

# data redistribution
dataset_dir = './datasets/'
os.mkdir(dataset_dir)
train_dir = dataset_dir + 'train/'
os.mkdir(train_dir)
train_img_dir = train_dir + 'images/'
os.mkdir(train_img_dir)
train_xml_dir = train_dir + 'xml/'
os.mkdir(train_xml_dir)
val_dir = dataset_dir + 'val/'
os.mkdir(val_dir)
val_img_dir = val_dir + 'images/'
os.mkdir(val_img_dir)
val_xml_dir = val_dir + 'xml/'
os.mkdir(val_xml_dir)

# for training set
for file_name in train_list:
    # for images
    from_file = from_img_dir + file_name
    print("0",from_file)
    to_file = train_img_dir + file_name
    print("1",to_file)
    shutil.copy(from_file, to_file)
    # for xml
    xml_name = file_name[:-4] + '.xml'
    from_file = from_xml_dir + xml_name
    to_file = train_xml_dir + xml_name
    print("2",to_file)
    print("3",from_file)
    shutil.copy(from_file, to_file)
    
# for validation set
for file_name in val_list:
    # for images
    from_file = from_img_dir + file_name
    to_file = val_img_dir + file_name
    shutil.copy(from_file, to_file)
    # for xml
    xml_name = file_name[:-4] + '.xml'
    from_file = from_xml_dir + xml_name
    to_file = val_xml_dir + xml_name
    shutil.copy(from_file, to_file)




