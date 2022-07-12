# -*- coding: UTF-8 -*-
'''
@Project ：code
@File    ：Data_Read.py ， 读取文件夹下txt文件中数据
@Author  ：leeqingshui
'''

import os

# 测试txt文件夹路径
test_dir_path = "F:\\SEMG_Mechanical_Arm\\code\\Data_Fitting\\Dataset\\train_dataset\\big_arm_01"

'''
@ 函数功能                  ：读取txt文件中每一行的数据
@ 入口参数 {str}  file_name ：txt文件路径
@ 返回参数 {list} data      ：存放每一行数据的列表
'''
def openreadtxt(file_name):
    data = []
    # 打开文件
    file = open(file_name,'r')
    # 读取所有行
    file_data = file.readlines()
    for row in file_data:
        # 去掉列表中每一个元素的换行符
        row = row.strip('\n')
        # 按‘，’切分每行的数据
        tmp_list = row.split(' ')
        # 将每行数据插入data列表中
        data.append(tmp_list)
    return data

'''
@ 函数功能                  ：读取根文件夹下面每一个文件夹中的txt文件
@ 入口参数 {str}  dir_path  ：文件夹路径
@ 返回参数 {list} txt_path  ：存放每一个txt文件路径的列表
'''
def opendirtxt(dir_path):
    txt_path = []

    # 遍历文件夹
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for file in filenames:
            if os.path.splitext(file)[1] == '.txt':
                txt_path.append(os.path.join(dirpath,file))
    return txt_path

'''
@ 函数功能                  ：每一个文件夹数据由 [['num']...[]]转换为[num,num,num...]
@ 入口参数 {list} txt_data  ：文件数据 , 格式：[['num']...[]]
@ 返回参数 {list} num_list  ：转换后电压数据，格式：[num,num,num...]
'''
def ListToNum(txt_data):
    V_list = []

    for line_data in txt_data:
        line_data = line_data[0]
        line_data = int(''.join([x for x in line_data if x.isdigit()])) / 1000000
        V_list.append(int(line_data))
    return V_list

if __name__=="__main__":
    # 读取文件夹下每一个子文件中txt路径
    # 保存到列表txt_path中
    txt_path = opendirtxt(test_dir_path)

    # 电压数据列表
    V_list = []

    # 遍历txt文件路径读取
    for file_path in txt_path:
        data = openreadtxt(file_path)
        V_list = ListToNum(data)
        print(V_list)
