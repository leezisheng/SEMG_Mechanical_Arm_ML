# -*- coding: UTF-8 -*-
'''
@Project ：code
@File    ：Data_Show.py ， 数据展示
@Author  ：leeqingshui
'''

from matplotlib import pyplot as plt
from Data_Read import openreadtxt,opendirtxt,ListToNum

'''
@ 函数功能                  ：绘出波形二维折线图
@ 入口参数 {list} num_list  ：转换后电压数据列表，格式：[num,num,num...]
'''
def Show_2D_Plot(num_list):

    Y_List = num_list
    X_List = list(range(len(Y_List)))

    # 设置标题
    plt.title('Emg waveform line diagram')

    # 设置坐标轴标签
    plt.xlabel('time')
    plt.ylabel('value(mv)')

    # 绘制折线图
    plt.plot(X_List, Y_List)

    # 绘制
    plt.show()

if __name__=="__main__":
    # 测试txt文件夹路径
    test_dir_path = "F:\\SEMG_Mechanical_Arm\\code\\Data_Fitting\\Dataset\\train_dataset\\big_arm_01"

    # 读取文件夹下每一个子文件中txt路径
    # 保存到列表txt_path中
    txt_path = opendirtxt(test_dir_path)

    # 电压数据列表
    V_list = []

    # 遍历txt文件路径读取
    for file_path in txt_path:
        data = openreadtxt(file_path)
        V_list = ListToNum(data)
        Show_2D_Plot(V_list)
        print(V_list)