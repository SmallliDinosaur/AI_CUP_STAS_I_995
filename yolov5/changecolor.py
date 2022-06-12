import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#from IPython.display import clear_output

def show_img(img):
#     plt.figure(figsize=(15,15)) 
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

def modify_color_temperature(img):
    
    # ---------------- 冷色調 ---------------- #  
    
#     height = img.shape[0]
#     width = img.shape[1]
#     dst = np.zeros(img.shape, img.dtype)

    # 1.計算三個通道的平均值，並依照平均值調整色調
    imgB = img[:, :, 0] 
    imgG = img[:, :, 1]
    imgR = img[:, :, 2] 

    # 調整色調請調整這邊~~ 
    # 白平衡 -> 三個值變化相同
    # 冷色調(增加b分量) -> 除了b之外都增加
    # 暖色調(增加r分量) -> 除了r之外都增加
    bAve = cv2.mean(imgB)[0] 
    gAve = cv2.mean(imgG)[0] + 20
    rAve = cv2.mean(imgR)[0] + 20
    aveGray = (int)(bAve + gAve + rAve) / 3

    # 2. 計算各通道增益係數，並使用此係數計算結果
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve
    imgB = np.floor((imgB * bCoef))  # 向下取整
    imgG = np.floor((imgG * gCoef))
    imgR = np.floor((imgR * rCoef))

    # 3. 變換後處理
#     for i in range(0, height):
#         for j in range(0, width):
#             imgb = imgB[i, j]
#             imgg = imgG[i, j]
#             imgr = imgR[i, j]
#             if imgb > 255:
#                 imgb = 255
#             if imgg > 255:
#                 imgg = 255
#             if imgr > 255:
#                 imgr = 255
#             dst[i, j] = (imgb, imgg, imgr)

    # 將原文第3部分的演算法做修改版，加快速度
    imgb = imgB
    imgb[imgb > 255] = 255
    
    imgg = imgG
    imgg[imgg > 255] = 255
    
    imgr = imgR
    imgr[imgr > 255] = 255
        
    cold_rgb = np.dstack((imgb, imgg, imgr)).astype(np.uint8) 
            
#     print("Cold color:")
#     print(cold_rgb.shape)
#     show_img(cold_rgb)



    # ---------------- 暖色調 ---------------- #  
    
     # 1.計算三個通道的平均值，並依照平均值調整色調
    imgB = img[:, :, 0] 
    imgG = img[:, :, 1]
    imgR = img[:, :, 2] 

    # 調整色調請調整這邊~~ 
    # 白平衡 -> 三個值變化相同
    # 冷色調(增加b分量) -> 除了b之外都增加
    # 暖色調(增加r分量) -> 除了r之外都增加
    bAve = cv2.mean(imgB)[0] + 20
    gAve = cv2.mean(imgG)[0] + 20
    rAve = cv2.mean(imgR)[0] 
    aveGray = (int)(bAve + gAve + rAve) / 3

    # 2. 計算各通道增益係數，並使用此係數計算結果
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve
    imgB = np.floor((imgB * bCoef))  # 向下取整
    imgG = np.floor((imgG * gCoef))
    imgR = np.floor((imgR * rCoef))

    # 3. 變換後處理
#     for i in range(0, height):
#         for j in range(0, width):
#             imgb = imgB[i, j]
#             imgg = imgG[i, j]
#             imgr = imgR[i, j]
#             if imgb > 255:
#                 imgb = 255
#             if imgg > 255:
#                 imgg = 255
#             if imgr > 255:
#                 imgr = 255
#             dst[i, j] = (imgb, imgg, imgr)

    # 將原文第3部分的演算法做修改版，加快速度
    imgb = imgB
    imgb[imgb > 255] = 255
    
    imgg = imgG
    imgg[imgg > 255] = 255
    
    imgr = imgR
    imgr[imgr > 255] = 255
        
    warm_rgb = np.dstack((imgb, imgg, imgr)).astype(np.uint8) 


#     print("Warm color:")
#     print(warm_rgb.shape)
#     show_img(warm_rgb)


    
    # ---------------- 印出結果圖表 ---------------- #  
    
    plt.figure(figsize=(15,15)) 
    plt.subplot(1, 3, 1)                 
    increase_img = cv2.cvtColor(cold_rgb, cv2.COLOR_BGR2RGB)
    #plt.imshow(increase_img)
    #plt.title("Cold color", {'fontsize':20})  

    plt.subplot(1, 3, 2)
    decrease_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(decrease_img)
    #plt.title("Origin picture", {'fontsize':20})
    
    plt.subplot(1, 3, 3)
    decrease_img = cv2.cvtColor(warm_rgb, cv2.COLOR_BGR2RGB)
    #plt.imshow(decrease_img)
    #plt.title("Warm color", {'fontsize':20})
    
    #plt.show()
    cv2.imwrite("./datasets/Private03/" + filename, increase_img)

def img_processing(img):
    # do something here
    modify_color_temperature(img)

directory_name='./datasets/Private03'
for filename in os.listdir(directory_name):
    img = cv2.imread(directory_name + "/" + filename)
    print(filename)
    # show_img(origin_img)

    result_img = img_processing(img)
    #show_img(result_img)
    #cv2.imwrite("./datasets/Private04/" + filename, result_img)