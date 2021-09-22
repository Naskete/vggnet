#  my-vggnet

简单的vgg神经网络，用于进行图像分类识别，参考vgg19

### vggnet.py
训练模型

### recongnize.py
识别图像，获取结果

### dataset
存储训练用的图片数据，这里采用的是[MNIST](http://yann.lecun.com/exdb/mnist)数据集中的手写数字0-4部分，目录结构为`dataset/label/image`

更改图片数据请按此结构放置图片数据

### model
存储训练模型，进行图像识别时从此目录读取模型

### figure
存储训练模型的结果loss与accuracy值对比图，用于对网络模型进行调整

### label
存储图像对应标签，进行图像识别时从此处读取标签