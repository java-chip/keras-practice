# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 22:02:09 2018

@author: watanabelab
"""

import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

base_dir = "C:\\Users\\watanabelab\\work\\keras-practice\\chapter5\\dogcat\\small_dataset"
train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# 水増しする画像を選択
img_path = fnames[3]

# 画像を読み込み, サイズを変更
img = image.load_img(img_path, target_size=(150, 150))

# 形状が(150, 150, 3)のNumpy配列に変換
x = image.img_to_array(img)

# (1, 150, 150, 3)に変形
x = x.reshape((1,) + x.shape)

# ランダムに変換した画像のバッチを生成する
# 無限ループとなるため, 何らかのタイミングでbreakする必要がある
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
    
plt.show()