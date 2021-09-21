from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# 初始化参数
INIT_LR = 0.001  # 学习率
EPOCH = 600  # 训练轮次
BITCH_SIZE = 30


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
    """
    :param width:
    :param height:
    :param depth:
    :param classes:
    :return:
    """
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


def draw_figure(his, epoch):
    """
    :param his:
    :param epoch:
    :return:
    """
    n = np.arange(0, epoch)
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(n, his.history['loss'], label='train_loss')
    plt.plot(n, his.history['val_loss'], label='val_loss')
    plt.plot(n, his.history['accuracy'], label='train_acc')
    plt.plot(n, his.history['val_accuracy'], label='val_acc')
    plt.title('training loss and accuracy')
    plt.xlabel('epoch')
    plt.ylabel('loss/accuracy')
    plt.legend()
    plt.savefig('image/loss-accuracy-vgg.png')


def train_model(models, train_set_x, test_set_x, train_set_y, test_set_y, epoch, bitch):
    print('[INFO]开始训练模型...')
    opt = optimizers.gradient_descent_v2.SGD(lr=INIT_LR, decay=INIT_LR / epoch)
    models.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

    # 训练结果
    h = models.fit(train_set_x, train_set_y, validation_data=(test_set_x, test_set_y), epochs=epoch, batch_size=bitch)
    print('[INFO]正在评估模型...')
    predictions = models.predict(test_set_x, batch_size=bitch)
    print(classification_report(test_set_y.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=lb.classes_))
    return h


def save(models, label, model_name, label_name):
    print('[INFO]保存模型...')
    models.save(model_name)
    f = open(label_name, 'wb')
    f.write(pickle.dumps(label))
    f.close()


if __name__ == '__main__':
    read_image_set(data, labels, 'dataset', 32, 64)
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)
    # random_state 与 read_image_set 中一致
    (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=32)

    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    test_y = lb.transform(test_y)

    model = build_model(width=64, height=64, depth=3, classes=len(lb.classes_))
    history = train_model(model, train_x, test_x, train_y, test_y, EPOCH, BITCH_SIZE)
    draw_figure(history, EPOCH)
    save(model, lb, 'model/number_model', 'label/lb_bgg.pickle')
