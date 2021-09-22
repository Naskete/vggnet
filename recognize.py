from keras.models import load_model
import pickle
import cv2 as cv


def get_data(filepath, size):
    """
    读取图片并处理
    :param filepath: string 图片目录
    :param size: 预处理图片大小，需要与训练时大小一致
    :return: image 返回预处理完成的图片
    """
    # 读取图片
    image = cv.imread(filepath)
    # 预处理
    image = cv.resize(image, (size, size))
    image = image.astype('float') / 255.0
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image


def predict_result(image, model_path, label_path):
    """
    预测结果
    :param image: 处理完成候的图片
    :param model_path: 要载入的模型的路径
    :param label_path: 标签路径
    :return: res 返回预测结果，标签：百分比概率
    """
    model = load_model(model_path)
    lb = pickle.loads(open(label_path, 'rb').read())
    # 预测结果
    predict = model.predict(image)
    # 获取对应标签
    index = predict.argmax(axis=1)[0]
    label = lb.classes_[index]
    # 得到结果和对应概率
    res = '{}:{:.2f}%'.format(label, predict[0][index] * 100)
    return res


if __name__ == '__main__':
    data = get_data('dataset/1/mnist_train_3.png', 64)
    result = predict_result(data, 'model/number_model', 'label/lb_bgg.pickle')
    print(result)
