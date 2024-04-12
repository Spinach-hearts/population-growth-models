# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:03:53 2023

@author: Lijim
"""

import numpy as np
from scipy import interpolate#三次样条插值
from scipy import optimize as op#依函数类型拟合
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False #解决负数坐标显示问题
plt.rcParams['savefig.dpi'] = 600 #图片像素
plt.rcParams['figure.dpi'] = 600 #分辨率



#拉格朗日插值
def lagrange_(x_,y_,x0_):
    """
    定义拉格朗日插值函数
    能处理预测节点为一个、多个的情况
    """
    L={}
    m=len(x0_)
    for k in range(m):#该部分处理需要预测的节点L[x0]=L0
        x0=x0_[k]
        n=len(x_)
        L0=0
        for i in range(n):#该部分得到需要预测节点的预测值L0
            l=1
            x1=x_[i]
            y1=y_[i]
            for j in range(n):#该部分得到基函数l1,l2
                x2=x_[j]
                if x1!=x2:
                    l*=(x0-x2)/(x1-x2)#会导致内存溢出，x值过大
            L0+=l*y1
        L[x0]=L0
    
    return L#返回结果为字典



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
x0=np.array([2020,2030])
#############################插值模型
#画图
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
#拉格朗日插值
yi_L=np.array(list(lagrange_(x_1,y_1,xi_3).values()))
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_L, label='多项式插值（拉格朗日插值）')
plt.xlabel("x（年份）",fontdict={"fontsize":12})
plt.ylabel("y（亿）",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("1949-2010年的人口增长的插值模型",fontdict={"fontsize":14})
plt.legend()
y0_L1=np.array(list(lagrange_(x_1,y_1,x0).values()))
print("拉格朗日插值基于1949-2010年数据预测2020人口数量{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_L1[0],y0_L1[1]))
plt.subplot(2,2,2)
#拉格朗日插值
yi_L=np.array(list(lagrange_(x_2,y_2,xi_4).values()))
plt.scatter(x_2, y_2, label='待拟合函数节点')
plt.plot(xi_4, yi_L, label='多项式插值（拉格朗日插值）')
plt.xlabel("x（年份）",fontdict={"fontsize":12})
plt.ylabel("y（亿）",fontdict={"fontsize":12})
plt.vlines(2030,6,24,'g','dashed',label='2030年')
plt.title("1949-2020年的人口增长的插值模型",fontdict={"fontsize":14})
plt.legend()
y0_L2=np.array(list(lagrange_(x_2,y_2,x0).values()))
print("拉格朗日插值基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y0_L2[1]))
print("拉格朗日插值模型误差为{:.2%}".format(np.abs((y0_L1[0]-y_2[-1])/y_2[-1])))
print("拉格朗日插值模型稳定性为{:.2%}".format(np.abs((y0_L2[1]-y0_L1[1])/y0_L1[1])))

plt.show()
#################################插值模型的优化，使用样条插值
#画图
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
#拉格朗日插值
yi_L=np.array(list(lagrange_(x_1,y_1,xi_3).values()))
#三次样条插值，调用函数
tck = interpolate.splrep(x_1,y_1)
yi_yangtiao = interpolate.splev(xi_3,tck,der=0)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_L, label='多项式插值（拉格朗日插值）')
plt.plot(xi_3, yi_yangtiao, label='三次样条插值')
plt.xlabel("x（年份）",fontdict={"fontsize":12})
plt.ylabel("y（亿）",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("1949-2010年的人口增长的样条插值模型",fontdict={"fontsize":14})
y0_s1=interpolate.splev(x0,tck,der=0)
print("三次样条插值基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_s1[0],y0_s1[1]))
plt.legend()
plt.subplot(2,2,2)
#拉格朗日插值
yi_L=np.array(list(lagrange_(x_2,y_2,xi_4).values()))
#三次样条插值，调用函数
tck = interpolate.splrep(x_2,y_2)
yi_yangtiao = interpolate.splev(xi_4,tck,der=0)
plt.scatter(x_2, y_2, label='待拟合函数节点')
plt.plot(xi_4, yi_L, label='多项式插值（拉格朗日插值）')
plt.plot(xi_4, yi_yangtiao, label='三次样条插值')
plt.xlabel("x（年份）",fontdict={"fontsize":12})
plt.ylabel("y（亿）",fontdict={"fontsize":12})
plt.vlines(2030,6,24,'g','dashed',label='2030年')
plt.title("1949-2020年的人口增长的样条插值模型",fontdict={"fontsize":14})
plt.legend()
y0_s2=interpolate.splev(x0,tck,der=0)
print("三次样条插值基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y0_s2[1]))
print("三次样条插值模型误差为{:.2%}".format(np.abs((y0_s1[0]-y_2[-1])/y_2[-1])))
print("三次样条插值模型稳定性为{:.2%}".format(np.abs((y0_s2[1]-y0_s1[1])/y0_s1[1])))

plt.show()

########################################比较两者的稳定性
#画图  多项式插值模型
plt.figure(figsize=(8,6))
#拉格朗日插值
yi_L_1=np.array(list(lagrange_(x_1,y_1,xi_3).values()))
#三次样条插值，调用函数
tck_1 = interpolate.splrep(x_1,y_1)
yi_yangtiao_1 = interpolate.splev(xi_3,tck_1,der=0)
#拉格朗日插值
yi_L_2=np.array(list(lagrange_(x_2,y_2,xi_4).values()))
#三次样条插值，调用函数
tck_2 = interpolate.splrep(x_2,y_2)
yi_yangtiao_2 = interpolate.splev(xi_4,tck_2,der=0)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_L_1, label='1949-2010多项式插值（拉格朗日插值）')
plt.plot(xi_3, yi_yangtiao_1, label='1949-2010三次样条插值')
plt.plot(xi_4, yi_L_2, label='1949-2020多项式插值（拉格朗日插值）')
plt.plot(xi_4, yi_yangtiao_2, label='1949-2020三次样条插值')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("增加数据对于多项式插值模型的影响",fontdict={"fontsize":14})
plt.legend()
plt.show()
#由图知样条插值在2020-2030区间，被夹在多项式插值（拉格朗日插值）之间，可见样条插值的稳定性远优于多项式插值，通过2020年节点值的预测，样条插值也优于多项式插值（拉格朗日插值），毫无疑问在多项式插值的方法中应选择样条插值。




