# -*- coding: UTF-8 -*-
'''
@Project ：code
@File    ：Data_Filter.py ， 数据滤波
@Author  ：leeqingshui
'''

import scipy.signal as signal
from Data_Read import openreadtxt,opendirtxt,ListToNum
from Data_Show import Show_2D_Plot

'''
@ 函数功能                  ：对信号进行中值滤波
@ 入口参数 {list} num_list  ：电压数据，格式：[num,num,num...]
@ 返回参数 {list} rst_list  ：中值滤波后电压数据，格式：[num,num,num...]
'''
def Midian_Filter(num_list):

    rst_list = []
    # 中值滤波
    rst_list = signal.medfilt(num_list, 11)

    return rst_list

if __name__=="__main__":
    # 测试txt文件夹路径
    test_dir_path = "F:\\SEMG_Mechanical_Arm\\code\\Data_Fitting\\Dataset\\train_dataset"

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
        V_list = Midian_Filter(V_list)
        Show_2D_Plot(V_list)
        print(V_list)
