# xml2txt

import numpy as np
from lxml import etree
import cv2
import os

def xml2txt(from_xml_path, to_txt_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    label_list = []
    for obj in root.findall('object'):
        #print(obj.find('name').text)
        label_list.append(obj.find('name').text)
        for bbox in obj.findall('bndbox'):
            # bb = [xmin, ymin, xmax, ymax]
            bb = []
            for pts in bbox.getchildren():
                bb.append(int(pts.text))
    label_set = list(set(label_list))
    return label_set

'''    
for bbox in tree.xpath('//bndbox'):
    # bbox = [xmin, ymin, xmax, ymax]
    for pts in bbox.getchildren():
        print(pts.text)
'''

# main
mode = 'train'
img_dir = './datasets/' + mode + '/images/'
xml_dir = './datasets/' + mode + '/xml/'
gt_dir = './datasets/' + mode + '/gt_img/'
if(not os.path.exists(gt_dir)):
    os.mkdir(gt_dir)
txt_dir = './datasets/' + mode + '/labels/'
if(not os.path.exists(txt_dir)):
    os.mkdir(txt_dir)

label_dict = {'Background':0, 'stas':1, 'STAS':1 }
color_dict = {'Background':(0,0,0), 'stas':(0,255,255), 'STAS':(0,255,255) }

img_list = os.listdir(img_dir)
xml_list = os.listdir(xml_dir)
n_file = len(xml_list)
for i in range(n_file):
    xml_name = xml_list[i]
    img_name = img_list[i]
    img = cv2.imread(img_dir+img_name)
    # xml sparse
    tree = etree.parse(xml_dir+xml_name)
    root = tree.getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    # go through every region
    txt_info = []
    for obj in root.findall('object'):
        label_name = obj.find('name').text
        for bbox in obj.findall('bndbox'):
            # bb = [xmin, ymin, xmax, ymax]
            bb = []
            for pts in bbox.getchildren():
                bb.append(int(pts.text))
        pt1 = (bb[0],bb[1]) # top-left corner
        pt2 = (bb[2],bb[3]) # bottom-right corner
        color_set = color_dict[label_name]
        # draw bbox
        cv2.rectangle(img,pt1,pt2,color_set,2)
        # yolo txt : [label_num, xc, yc, w, h]
        label_num = label_dict[label_name]
        xc = ((bb[0]+bb[2])/2)/width
        yc = ((bb[1]+bb[3])/2)/height
        w = (bb[2]-bb[0])/width
        h = (bb[3]-bb[1])/height
        txt_info.append([label_num, xc, yc, w, h])
    txt_np = np.array(txt_info)
    txt_name = xml_name[:-4] + '.txt'
    np.savetxt(txt_dir+txt_name, txt_np, fmt='%1.3f', delimiter =' ')
    cv2.imwrite(gt_dir+img_name, img)




