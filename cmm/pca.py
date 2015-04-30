# ecoding=utf8
'''
Created on 2015-4-28

@author: qq
'''
import os, sys
from PIL import Image
import numpy
from matplotlib import pylab
from numpy import dot, linalg, sqrt, array


def pca(X):
    """ 主成分分析：
    输入：矩阵X ，其中该矩阵中存储训练数据，每一行为一条训练数据
    返回：投影矩阵（按照维度的重要性排序）、方差和均值"""
    # 获取维数
    num_data, dim = X.shape
    # 数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X
    if dim > num_data:
        # PCA- 使用紧致技巧
        M = dot(X, X.T)
        # 协方差矩阵
        e, EV = linalg.eigh(M)  # 特征值和特征向量
        tmp = dot(X.T, EV).T  # 这就是紧致技巧
        V = tmp[::-1]  # 由于最后的特征向量是我们所需要的，所以需要将其逆转
        S = sqrt(e)[::-1]  # 由于特征值是按照递增顺序排列的，所以需要将其逆转
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA- 使用SVD 方法
        U, S, V = linalg.svd(X)
        V = V[:num_data]  # 仅仅返回前nun_data 维的数据才合理
        # 返回投影矩阵、方差和均值
    return V, S, mean_X

img = Image.open("Chrysanthemum.jpg").convert('L')
h, l = img.size
img.save("Chrysanthemum.jpg_L.jpg")
imgs= Image.open("Desert.jpg").convert('L')

imgss= Image.open("Hydrangeas.jpg").convert('L')

h, l = imgs.size
immatrix = array([array(imgs).flatten(), array(imgss).flatten(), array(img).flatten(), array(imgs).flatten()])



V, S, immean = pca(immatrix)
img1 = V[0].reshape(l, h) 

#pylab.subplot(2,4,1)

pylab.imshow(img1)
#pylab.subplot(2,4,2)
#pylab.imshow(immean.reshape(l, h) )
pylab.gray()  
pylab.show()  
