# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:13:56 2023

@author: Lijim
"""

import numpy as np
from scipy import interpolate#三次样条插值
from scipy import optimize as op#依函数类型拟合
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率


def xianxinnihe(x_,y_,xi):
    #最小二乘法线性拟合
    p0_p0=len(x_)
    p0_p1=x_.sum()
    p1_p1=np.dot(x_,x_)
    y_ln=np.log(y_)
    p0_w=y_ln.sum()
    p1_w=np.dot(x_,y_ln)

    A=np.array([[p0_p0,p0_p1],[p0_p1,p1_p1]])# A为系数矩阵
    b=np.array([p0_w,p1_w])# b为常数列
    #求解线性方程组
    inv_A = np.linalg.inv(A)  # A的逆矩阵
    x = inv_A.dot(b)  # A的逆矩阵与b做点积运算
    x = np.linalg.solve(A, b) # 5,6两行也可以用本行替代
    a=np.exp(x[0])
    b=x[1]
    f=lambda xi:a*np.exp(b*xi)
    y_zhishu=f(xi)
    print("由最小二乘得到的拟合函数为：y={}*exp({:.3f}x)".format(a,b))
    print(a*np.exp(b*2020))
    return y_zhishu



#已知条件
#中国人口
x_1=np.array([1949,1953,1965,1982,1990,2000,2010])
y_1=np.array([5.42,5.88,7.25,10.17,11.43,12.67,13.40])
x_2=np.array([1949,1953,1965,1982,1990,2000,2010,2020])
y_2=np.array([5.42,5.88,7.25,10.17,11.43,12.67,13.40,14.43])
xi_1=np.linspace(1949,2020,100)
xi_2=np.linspace(1949,2030,100)
xi_3=range(1949,2023,1)
xi_4=range(1949,2033,1)


###############################模型对比
###########################插值模型与指数模型
#画图
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
#三次样条插值，调用函数
tck = interpolate.splrep(x_1,y_1)
yi_yangtiao = interpolate.splev(xi_3,tck,der=0)
#指数模型的最小二乘拟合
y_nihe_power=xianxinnihe(x_1,y_1,xi_3)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_yangtiao, label='三次样条插值')
plt.plot(xi_3, y_nihe_power, label='指数模型的最小二乘拟合')
plt.xlabel("x（年份）",fontdict={"fontsize":12})
plt.ylabel("y（亿）",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("1949-2010年的人口增长的样条插值模型",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,2,2)
#三次样条插值，调用函数
tck = interpolate.splrep(x_2,y_2)
yi_yangtiao = interpolate.splev(xi_4,tck,der=0)
#指数模型的最小二乘拟合
y_nihe_power=xianxinnihe(x_2,y_2,xi_4)
plt.scatter(x_2, y_2, label='待拟合函数节点')
plt.plot(xi_4, yi_yangtiao, label='三次样条插值')
plt.plot(xi_4, y_nihe_power, label='指数模型的最小二乘拟合')
plt.xlabel("x（年份）",fontdict={"fontsize":12})
plt.ylabel("y（亿）",fontdict={"fontsize":12})
plt.vlines(2030,6,24,'g','dashed',label='2030年')
plt.title("1949-2020年的人口增长的样条插值模型",fontdict={"fontsize":14})
plt.legend()
plt.show()





