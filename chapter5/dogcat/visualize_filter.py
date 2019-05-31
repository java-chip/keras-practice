#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:13:57 2019

@author: tsugaike3
"""

import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import backend as K
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

#layer_name ='block3_conv1'
#filter_index = 0
    
# テンソルを有効な画像に変換するユーティリティ関数
def deprocess_image(x):
    # テンソルを正規化: 中心を0, 標準偏差を0.1にする
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    # [0, 1]でクリッピング
    x += 0.5
    x = np.clip(x, 0, 1)
    
    # RGB配列に変換
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    # フィルタを可視化するための損失テンソルの定義
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    
    # 入力に関する損失関数の勾配を取得
    # gradientsの呼び出しはテンソルのリストを返す
    # このため、最初の要素(テンソル)だけを保持する
    grads = K.gradients(loss, model.input)[0]
    
    # 勾配の正規化
    # 除算の前に1e-5を足すことで、0による除算を回避
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    # 入力値をNumPy配列で受け取り, 出力値をNumpy配列で返す関数
    iterate = K.function([model.input], [loss, grads])
    
    # 確率的勾配降下法を使って損失値を最大化
    # 最初はノイズが含まれたグレースケール画像を適用
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    
    # 勾配上昇法を40ステップ実行
    step = 1.
    for i in range(40):
        # 損失値と勾配値を計算
        loss_value, grads_value = iterate([input_img_data])
        # 損失が最大になる方向に入力画像を調整
        input_img_data += grads_value * step    
    
    img = input_img_data[0]
    return deprocess_image(img)

#plt.imshow(generate_pattern('block3_conv1', 0))
#plt.show()

layers = ['block3_conv1']
for layer_name in layers:
    size = 64
    margin = 5
    
    # 結果を格納する空(黒)の画像
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):# resultsグリッドの行を順番に処理
        for j in range(8):# resultsグリッドの列を順番に処理
            
            # layer_nameのフィルタi + (j * 8)のパターンを生成
            filter_img = generate_pattern(layer_name, i + (j * 8), size = size)
            # resultsグリッドの矩形(i, j)に結果を配置
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                    vertical_start: vertical_end, :] = filter_img
    plt.figure(figsize=(20, 20))
    plt.imshow(results.astype('int'))
    plt.show()









