#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 01:53:58 2019

@author: tsugaike3
"""

import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import backend as K
import numpy as np
import cv2

# 出力側に全結合分類器が含まれていることに注意
# ここまでのケースでは、この分類器を削除している
model = VGG16(weights='imagenet')

# ターゲット画像へのローカルパス
img_path = '/home/tsugaike3/matsuda/work/keras-practice/chapter5/elephant.jpg'

# ターゲット画像を読み込む: imgはサイズが224x224のPIL画像
img = image.load_img(img_path, target_size=(224, 224))

# xは形状が(224, 224, 3)のfloat32型のNumPy配列
x = image.img_to_array(img)

# この配列をサイズが(1, 224, 224, 3)のバッチに変換するために次元を追加
x = np.expand_dims(x, axis=0)

# バッチの前処理(チャネルごとに色を正規化)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
print(np.argmax(preds[0]))

#予測ベクトルの「アフリカゾウ」エントリ
african_elephant_output = model.output[:, 386]

#VGG16の最後の畳み込み層であるblock5_conv3の出力特徴マップ
last_conv_layer = model.get_layer('block5_conv3')

#block5_conv3の出力特徴マップでの「アフリカゾウ」クラスの勾配
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

#形状が(512, )のベクトル
#各エントリは特定の特徴マップチャネルの勾配の平均強度
pooled_grads = K.mean(grads, axis=(0, 1, 2))

#2頭のアフリカゾウのサンプル画像に基づいて、pooled_gradsと
#block5_conv3の出力特徴マップの値にアクセスするための関数
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])

#これら2つの値をNumPy配列として取得
pooled_grads_value, conv_layer_output_value = iterate([x])

#「アフリカゾウ」クラスに関する「このチャネルの重要度」を
#特徴マップ配列の各チャネルにかける
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

#最終的な特徴マップのチャネルごとの平均値が
#クラスの活性化のヒートマップ
heatmap = np.mean(conv_layer_output_value, axis = -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# ヒートマップをRGBに変換
heatmap = np.uint8(255 * heatmap)

# ヒートマップを元の画像に適用
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4はヒートマップの強度係数
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('/home/tsugaike3/matsuda/work/keras-practice/chapter5/elephant_cam.jpg', superimposed_img)



























