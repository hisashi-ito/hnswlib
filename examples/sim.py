#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 【sim】
#
#  概要:
#        cosine の similarity を計算する
#
import hnswlib
import numpy as np

# データ作成 (1000, 5)
num_elements = 10000
dim = 100
data = np.float32(np.random.randint(0, 10, (num_elements, dim)))

# データを分割する
# (500, 16), (500, 16)
data1 = data[:num_elements // 2]
data2 = data[num_elements // 2:]
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=num_elements//2, ef_construction=100, M=16)
index.set_num_threads(4)
index.add_items(data1)

# cosine だからベクトルの正規化(norm)で割り算している点もO.K.
for i, v in enumerate(data1):
    labels, distances = index.knn_query(v, 30)
    scores = 1 - distances
    print("index: {} label {} score: {}".format(i, labels[0][0], scores[0][0]))
