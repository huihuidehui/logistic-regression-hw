#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def binary_value(column_data,cond):
    '''
    args:
        column_data: 需要二值化的列
        cond: 二值化的条件，符合该条件的设置为1否则为0
    return: 返回处理好的数据
    '''
    column_data.mask(cond=cond,other=1,inplace=True)
    column_data.where(cond=column_data==1,other=0,inplace=True)
    #print(column_data)
    return column_data.astype('int64')

def one_hot_scale(data):
    '''
    对数据进行one-hot编码，并进行缩放.
    args:
        data: 原始数据
    return: 处理后的数据
    '''
    object_columns = [column for column in data.columns if data[column].dtype == 'object']
    number_columns = [column for column in data.columns if data[column].dtype == 'int64' and column != 'income' and column != 'sex']
    object_columns, number_columns = data[object_columns],data[number_columns]  # 提取相应的列
    
    # 对数据进行标准化缩放,缩放后值在0-1之间
    number_columns = (number_columns - number_columns.min()) / (number_columns.max() - number_columns.min())
    object_columns = pd.get_dummies(object_columns)
    data = pd.concat([object_columns, number_columns, data['sex'], data['income']],axis=1)
    
    return data

def data_pre_process(df_data,df_test_data):
    '''
    1. 进行数据预处理
    2. 返回训练集和验证集
    '''
    df_data,df_test_data = df_data.fillna(0),df_test_data.fillna(0)  # 对空值进行填充

    # 对性别和income列处理
    df_data['sex'] = binary_value(df_data['sex'].copy(deep=True),cond=df_data['sex']==' Male')
    df_test_data['sex'] = binary_value(df_test_data['sex'].copy(deep=True),cond=df_test_data['sex']==' Male')
    df_data['income'] = binary_value(df_data['income'].copy(deep=True),cond=df_data['income']==' >50K')

    df_data = one_hot_scale(df_data)
    df_test_data= one_hot_scale(df_test_data)

    # 把训练集分为训练集和验证集
    x_train = df_data.iloc[:int(len(df_data.index) * 0.80), :]
    x_valid = df_data.iloc[int(len(df_data.index) * 0.80):, :]
    
    y_train = np.array(x_train.iloc[:, -1:])
    x_train = np.array(x_train.iloc[:, :-1])
    x_train = np.c_[x_train, np.ones((x_train.shape[0], 1))]  # 添加一个横为1的列，用于简化偏置系数b的更新
    y_valid = np.array(x_valid.iloc[:, -1:])
    x_valid = np.array(x_valid.iloc[:, :-1])
    x_valid = np.c_[x_valid, np.ones((x_valid.shape[0], 1))]
    
    y_test = np.array(df_test_data.iloc[:,-1:])
    x_test = np.array(df_test_data.iloc[:,:-1])
    x_test = np.c_[x_test, np.ones(
        (x_test.shape[0], 1))]  # 添加一个横为1的列，用于简化偏置系数b的更新
    return x_train, y_train, x_valid, y_valid,x_test,y_test


def gradient_secent(x_train, y_train):
    '''
    args:
        x_train: m x n
        y_train: m x 1
    return: wb

    '''
    lr_rate, threshold = 1, 5
    wb = np.ones((x_train.shape[1], 1))  # n x 1
    g = np.zeros((x_train.shape[1], 1))
    y_predict = get_y_predict(wb, x_train)
    loss_value = get_loss_value(wb, y_train, y_predict)  # 计算损失值
    loss_value_list = []
    while True:
        loss_value_list.append(loss_value)
        wb, g = update_parameter(wb, lr_rate, x_train, y_train, y_predict, g)
        y_predict = get_y_predict(wb, x_train)
        pre_loss_value, loss_value = loss_value, get_loss_value(
            wb, y_train, y_predict)
        if abs(pre_loss_value - loss_value) < threshold: break
    plot_loss_value(loss_value_list)
    return wb


def update_parameter(wb, lr_rate, x_train, y_train, y_predict, g):
    '''
    args:
        wb: n x 1
        g: adagrad 里的 G, n x 1
        lr_rate: 学习速率
        y_train: 真实值，m x 1
        y_predict: 预测值，m x 1
    return: wb, g
    '''
    descent = x_train.T.dot(y_predict - y_train)  # n x 1
    g = g + descent * descent
    lr_rates = lr_rate / (np.sqrt(g + 0.00000001))  # n x 1
    wb = wb - lr_rates * descent
    return wb, g


def get_y_predict(wb, x_train):
    '''
    args:
        wb: 参数，n x 1
        x_train: 训练集，m x n
    return:计算预测值 y.shape = m x 1
    '''
    y = wb.T.dot(x_train.T)  # 1 x n * n x m = 1 x m
    y = 1 / (np.exp(-1 * y) + 1)  # 1 x m
    y = y.T  # m x 1
    return y


def get_loss_value(wb, y_train, y_predict):
    '''
    args:
        wb: 参数，n x 1
        y_train: 真实值，m x 1
        y_predict: 预测值，m x 1
    return: loss_value, 损失值
    '''
    res = y_train.T.dot(np.log(y_predict))  # 1 x m * m x 1
    res = -1 * res
    tmp = (y_train - 1).T.dot(np.log(1 - y_predict))
    res = res + tmp
    return res[0][0]


def get_acc(wb, x, y):
    '''
    args:
        wb: n x 1
        x: m x n
        y: m x 1
    return: 正确率 acc
    '''
    y_predict = get_y_predict(wb, x)  # m x 1
    y_predict[y_predict > 0.5] = 1
    y_predict[y_predict <= 0.5] = 0
    acc = sum(y_predict == y)[0] / y.shape[0]
    return acc

def test_data_process(df_data):
    '''
    args:
        df_data: m x n
    return:
        x_test, y_test
    '''

def main():
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    x_train, y_train, x_valid,y_valid,x_test,y_test = data_pre_process(train_data,test_data)
    wb = gradient_secent(x_train, y_train)
    valid_acc = get_acc(wb, x_valid, y_valid)
    print("valid accuracy:{}".format(valid_acc))
    test_acc = get_acc(wb,x_test,y_test)
    print("tet accuracy:{}".format(test_acc))
    plt.show()

def plot_loss_value(loss_values):
    '''
    画出loss value的变化图
    '''
    plt.plot(range(len(loss_values)),loss_values,linewidth=4,color='g')

if __name__ == '__main__':
    '''
    李宏毅机器学习第二次作业
    1. 对数几率回归
    2. 使用梯度下降寻找最优解
    '''
    print("正在执行中，可能需要耗费大量时间，如果超过10分钟未响应请手动关掉即可。")
    main()
