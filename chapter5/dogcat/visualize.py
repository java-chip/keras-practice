#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 00:33:45 2019

@author: tsugaike3
"""

# 中間層の出力を可視化するプログラム
from keras import models
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('cats_and_dogs_small_2.h5')
model.summary()

img_path = '/home/tsugaike3/matsuda/work/keras-practice/chapter5/dogcat/small_dataset/test/cats/cat.1500.jpg'

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

# 出力側の8つの層から出力を抽出
layer_outputs = [layer.output for layer in model.layers[:8]]

# 特定の入力をもとに、これらの出力を返すモデルを作成
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 5つのNumPy配列(層の活性化ごとに1つ)のリストを返す
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# 3番目のチャネルを可視化
plt.matshow(first_layer_activation[0, :, :, 3], cmap = 'viridis')
plt.show()

plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()




































