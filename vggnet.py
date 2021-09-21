from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
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


def read_image_set(dataset, label, catalogue, random_state, size=64):
    """
    :param dataset:
    :param label:
    :param catalogue:
    :param random_state:
    :param size:
    :return:
    """
    print('[INFO]start reading image data...')
    root_dir = os.listdir(catalogue)
    for dirs in root_dir:
        path = catalogue + '/' + dirs
        file = os.listdir(path)[:10]
        # 每次读取数据顺序随机
        random.seed(random_state)
        random.shuffle(file)
        for filename in file:
            target = path + '/' + filename
            # 读取图片，并进行预处理
            image = cv.imread(target)
            image = cv.resize(image, (size, size))
            dataset.append(image)
            # 获取对应标签，即文件夹的名字
            lb = target.split(os.path.sep)[-2]
            label.append(lb)


def build_model(width, height, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)
    dim = -1

    # 添加卷积层
    # (CONV => RELU)*2 => POOL
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # (CONV => RELU) * 4 => POOL
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # (CONV => RELU) * 4 => POOL
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # (CONV => RELU) * 4 => POOL
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(axis=chan_dim))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # FC 层
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # 分类
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    read_image_set(data, labels, 'dataset', 32)
