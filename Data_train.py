# -*- coding: UTF-8 -*-
'''
@Project ：code
@File    ：Data_Filter.py ， 数据训练
@Author  ：leeqingshui
'''
import numpy

from Data_Read import openreadtxt,opendirtxt,ListToNum
from Data_Filter import Midian_Filter

import numpy as np
import keras
from keras.layers import Dense, Conv1D ,Flatten , MaxPool1D, Dropout, Activation

from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

dataset_path = "F:\\SEMG_Mechanical_Arm\\code\\Data_Fitting\\Dataset\\train_dataset"

'''
@ 函数功能                  ：加载数据
@ 入口参数 {list} txt_data  ：文件数据 
@ 返回参数 {list} num_list  ：转换后电压数据，格式：[num,num,num...]
@ 返回参数 {list} num_list  ：动作标签，格式：0或1，其中0代表没有动作；1代表有动作
'''
def load_data(dataset_path):
    txt_path = []

    # 训练数据列表
    train_data = []
    # 标签列表
    label_list = []
    # 文件夹命名列表
    seq_names = ['arm_move','arm_no_move']

    # 读取文件夹下每一个子文件中txt路径
    # 保存到列表txt_path中
    txt_path = opendirtxt(dataset_path)
    # # 遍历txt文件路径读取
    for file_path in txt_path:
        if seq_names[0] in file_path:
            # print(file_path)
            # 滤波+字符串转数字+长度截取
            train_data.append(np.array(Midian_Filter(ListToNum(openreadtxt(file_path)))[0:150]))
            label_list.append(1)
        else:
            train_data.append(np.array(Midian_Filter(ListToNum(openreadtxt(file_path)))[0:150]))
            label_list.append(0)

    # print(np.array(train_data))
    # print(np.array(label_list))

    return np.array(train_data), np.array(label_list)

if __name__=="__main__":

    # 加载数据
    train_data,train_label = load_data(dataset_path)
    print(train_data.shape, train_label.shape)

    # 增加一维轴
    train_data = train_data.reshape(-1,16,150)
    print(train_data.shape)
    # 对标签进行独热编码

    lb = LabelBinarizer()
    train_label = lb.fit_transform(train_label)  # transfer label to binary value
    train_label = numpy.array(to_categorical(train_label))  # transfer binary label to one-hot. IMPORTANT
    # train_label = np.expand_dims(train_label, axis=0)
    print(train_label.reshape(16,2))

    # 构建顺序模型
    model = keras.models.Sequential()

    # 卷积层1 + relu
    # 25 卷积核的数量 即输出的维度
    # 3 每个过滤器的长度
    model.add(Conv1D(32, 3, activation='relu', input_shape=(16, 150), padding="same"))
    # 池化层1
    model.add(MaxPool1D(pool_size=3, strides=3))

    # 卷积层2 + relu
    model.add(Conv1D(64, 3, strides=1, activation='relu', padding='same'))
    # 池化层2
    model.add(MaxPool1D(pool_size=3, strides=3))

    # 神经元随机失活
    model.add(Dropout(0.25))
    # 拉成一维数据
    model.add(Flatten())
    # 全连接层1
    model.add(Dense(1024))
    # 激活层
    model.add(Activation('relu'))

    # 随机失活
    model.add(Dropout(0.4))
    # 全连接层2
    model.add(Dense(2))
    # Softmax评分
    model.add(Activation('softmax'))

    # 查看定义的模型
    model.summary()

    # 优化器配置
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练
    history = model.fit(train_data, train_label, epochs=10, batch_size=1,verbose=1)

    print(history.params)

    model.save("model.h5")












