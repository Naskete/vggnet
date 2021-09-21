from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten, BatchNormalization
import keras.losses
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import cv2 as cv
import os

data = []
labels = []


def read_image_set(dataset, label, catalogue):
    print('[INFO]start reading image data...')
    root_dir = os.listdir(catalogue)
    for dirs in root_dir:
        path = catalogue + '/' + dirs
        file = os.listdir(path)[:10]
        # 每次读取数据顺序随机
        random.seed(32)
        random.shuffle(file)
        for filename in file:
            target = path + '/' + filename
            # 读取图片，并进行预处理
            image = cv.imread(target)
            image = cv.resize(image, (64, 64))
            dataset.append(image)
            # 获取对应标签，即文件夹的名字
            lb = target.split(os.path.sep)[-2]
            label.append(lb)


if __name__ == '__main__':
    read_image_set(data, labels, 'dataset')
