# xml2txt

import numpy as np
from lxml import etree
import os

def xml_label_check(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()

    label_list = []
    for obj in root.findall('object'):
        #print(obj.find('name').text)
        label_list.append(obj.find('name').text)

    label_set = list(set(label_list))
    
    return label_set

# main

xml_dir = './OBJ_Train_Datasets/Train_Annotations/'
label_collection = []
for xml_file in os.listdir(xml_dir):
    label_set = xml_label_check(xml_dir+xml_file)
    label_collection.extend(label_set)

label_collection = list(set(label_collection)) 
print(label_collection)




