![imgur](https://i.imgur.com/4O2X2RO.jpeg)
![imgur](https://i.imgur.com/EaKUhne.png)

# AI_CUP_STAS_I_995
肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 I：運用物體偵測作法於找尋STAS
報告說明文件
<h1>壹、	環境</h1>
　　我們使用Windows作業系統來進行這個專案，並且以Python開發環境，但這個環境不見得可以支援GPU加速計算，雖說Pytorch也是支援CPU計算，但單使用CPU計算時間會非常久，所以我們運用CPU Intel(R) Core(TM) i9-11900H @ 2.50GHz搭配GPU NVIDIA GEFORCE RTX 3080 RTX來加速計算。使用github上ultralytics 所提供Pytorch 框架的 yolov5，是一種基於yolov4 : AlexexAB fork github的目標檢測算法。

* 作業系統：Windows
* CPU：Intel(R) Core(TM) i9-11900H @ 2.50GHz
* GPU：NVIDIA GEFORCE RTX 3080 RTX
* 語言：Python 3.8.10
* Torch：torch 1.9.0＋cu111
* yolov5 : ultralytics fork github 
<h1>貳、	演算方法與模型架構</h1>
　　使用GITHUB上，ultralytics所提供的yolov5。yolov5官方程式碼中，給出的目標檢測網路中一共有4個版本，分別是yolov5s、yolov5m、yolov5l、yolov5x四個模型。yolov5程式碼中給出的網路檔案是yaml格式檔案，和原本yolov3、yolov4中的cfg格式檔案不同。yolov5作者是在COCO資料集上進行的測試，而 COCO資料集的小目標佔比，因此最終的四種網路結構，效能上來說各有千秋。yolov5s網路複雜度最小，速度最快，AP精度也最低。其他的三種網路，在此基礎上，不斷加深加寬網路，AP精度也不斷提升，但速度的消耗也在不斷增加。圖一是yolov5作者的演算法效能測試圖，表一是預訓練的檢查點，所有檢查點都使用預設設置訓練到 300 個 epoch。

* Nano 和 Small 模型使用hyp.scratch-low.yaml hyps，所有其他模型使用hyp.scratch-high.yaml。mAP val值適用於COCO val2017數據集上的單模型單尺度。
```python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65```

* TTA 測試時間增強包括反射和比例增強。
```python val.py --data coco.yaml --img 1536 --iou 0.7 --augment```

![Imgur](https://i.imgur.com/kI9Vu4F.png "yolov5作者的演算法效能測試圖")[1]
![Imgur](https://i.imgur.com/ugNm0x5.png "預訓練的檢查點")[1]

* 網址：https://github.com/ultralytics/Yolov5/releases
* 使用模型包括：YOLOv5n、YOLOv5s、YOLOv5m、YOLOv5l、YOLOv5x、YOLOv5n6、YOLOv5s6、YOLOv5m6、YOLOv5l6、YOLOv5x6+ TTA。
<h1>參、	資料處理</h1>
資料預處理部分，主要做三個動作。
利用主辦單位所提供的壓縮檔(OBJ_Train_Datasets.zip)資訊，其中內容分別有Train_Annotations、Train_Images兩個資料夾，內含1053張圖片(jpg)以及1053個標記檔(xml)。

* 步驟一：採用隨機分配固定比例方式分成8比2，訓練集842筆和測試集131筆。附檔檔名：Step_1_data_split.py
```
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
    hutil.copy(from_file, to_file)
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
```

* 步驟二：檢查xml檔。附檔檔名：Step_2_xml_check.py
```
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
```

* 步驟三：xml檔轉成程式所需的txt檔。附檔檔名：Step_3_xml2txt.py
```
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
```


<h1>肆、	訓練方式</h1>

　　用隨機分配固定比例方式分成8比2，訓練集842筆和測試集131筆。以表一權重檔做為基底，再加上新的資料繼續做訓練，期以能夠提高準確率。我們用上述的訓練方式重複訓練，有微調模型架構裡的超參數，訓練超參數包括：yaml文件的選擇，和訓練圖片的大小、預訓練、batch、epoch等。可以直接在train.py的parser中修改，也可以在命令行執行時修改，如下：[2]

```$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64```

* --data指定訓練數據文件
* --cfg設置網絡結構的配置文件
* --weights加載預訓練模型的路徑
* yaml文件：

```
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Hyperparameters for Objects365 training
# python train.py --weights yolov5m.pt --data Objects365.yaml --evolve
# See Hyperparameter Evolution tutorial for details https://github.com/ultralytics/yolov5#tutorials
lr0: 0.00258
lrf: 0.17
momentum: 0.779
weight_decay: 0.00058
warmup_epochs: 1.33
warmup_momentum: 0.86
warmup_bias_lr: 0.0711
box: 0.0539
cls: 0.299
cls_pw: 0.825
obj: 0.632
obj_pw: 1.0
iou_t: 0.2
anchor_t: 3.44
anchors: 3.2
fl_gamma: 0.0
hsv_h: 0.0188
hsv_s: 0.704
hsv_v: 0.36
degrees: 180.0 
translate: 0.0902
scale: 0.491
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
```

<h1>伍、	分析與結論</h1>

![Imgur](https://i.imgur.com/Ocs8Yn5.png "訓練過程")

<h1>陸、	雲端使用</h1>
 
　　比賽期間的Public成績，剛好很幸運在排名第11名，擁有台智雲將提供前30名的隊伍於5/26~6/1共7天等值新台幣3萬額度的TWCC雲端運算資源使用權限，換算可讓每個隊伍可以使用一張NVIDIA® Tesla V100 GPU約490個小時之雲端資源。
　　運用TWCC雲端運算資源的容器運算服務 (Container Compute Service, CCS)，他的服務項目包含開發型容器、任務型容器。配備 8 個 NVIDIA® Tesla V100 GPU，加速人工智慧訓練、推論與高效能運算，支援 5120 個 CUDA 核心與 640 個 Tensor 核心，並支援 NVLink 進行 GPU 之間的資料傳輸，加速人工智慧訓練、推論與高效能運算。[3]

![Imgur](https://i.imgur.com/nYfJdfu.png "競賽說明")

我們使用容器型號c.4xsuper進行運算，比較使用本機端CPU Intel(R) Core(TM) i9-11900H @ 2.50GHz搭配GPU NVIDIA GEFORCE RTX 3080 RTX，訓練時長差距三倍。
<h1>柒、	程式碼</h1>
我們將程式上架到GITHUB上，

* 網址：https://github.com/SmallliDinosaur/AI_CUP_STAS_I_995

1.	前處理程式碼。
* 網址：https://github.com/SmallliDinosaur/AI_CUP_STAS_I_995/tree/main/Before
2.	訓練程式碼。
3. 被訓練好的模型檔案(模型權重檔, 例如.mat格式)。
4. 辨識程式碼(包含上傳競賽網頁之預測結果輸出)。
5. 各項參數之設定。
6. 執行環境(所使用的程式版本(tensorflow/keras/pyrotch/matlab)與開發所需的額外Support package版本)。


<h1>捌、	使用的外部資源與參考文獻</h1>
[1]	ultralytics(2022, February 2). Yolov5x6.pt, Pretrained Checkpoints,Retrieved from https://github.com/ultralytics/Yolov5/releases

[2]	Laughing-q (2020, July 20). YOLOV5訓練代碼train.py註釋與解析, Retrieved from https://blog.csdn.net/Q1u1NG/article/details/107463417

[3]	TWS (2022). CCS 容器運算服務, Retrieved from https://tws.twcc.ai/service/container-compute-service/



 
<h1>聯絡資料</h1>

隊伍

* 隊伍名稱	Team 995

隊員

* 陳冠霖(Guan-Lin Chen)
* 吳貞儀(Chen-Yi Wu)

指導教授

* 徐位文


