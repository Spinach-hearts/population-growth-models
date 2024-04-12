# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:26:57 2023

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




#已知条件
#中国人口
x_1=np.array([1949,1953,1965,1982,1990,2000,2010])
y_1=np.array([5.42,5.88,7.25,10.17,11.43,12.67,13.40])
x_2=np.array([1949,1953,1965,1982,1990,2000,2010,2020])
y_2=np.array([5.42,5.88,7.25,10.17,11.43,12.67,13.40,14.43])
y_1ln=np.log(y_1)
y_2ln=np.log(y_2)
xi_1=np.linspace(1949,2020,100)
xi_2=np.linspace(1949,2030,100)
xi_3=range(1949,2023,1)
xi_4=range(1949,2033,1)
x0=np.array([2020,2030])



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

##########################指数模型
#画图
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
y_nihe_power=xianxinnihe(x_1,y_1,xi_3)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, y_nihe_power, label='指数模型的最小二乘拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("指数模型的最小二乘拟合",fontdict={"fontsize":14})
plt.legend()

plt.subplot(1,2,2)
y_nihe_power_1=xianxinnihe(x_1,y_1,xi_3)
y_nihe_power_2=xianxinnihe(x_2,y_2,xi_4)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, y_nihe_power_1, label='1949-2010指数模型')
plt.plot(xi_4, y_nihe_power_2, label='1949-2020指数模型')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("增加数据对于指数模型的影响",fontdict={"fontsize":14})
plt.legend()
y0_exp1=xianxinnihe(x_1,y_1,x0)
print("指数拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_exp1[0],y0_exp1[1]))
y0_exp2=xianxinnihe(x_2,y_2,x0)
print("指数拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y0_exp2[1]))
print("指数拟合模型误差为{:.2%}".format(np.abs((y0_exp1[0]-y_2[-1])/y_2[-1])))
print("指数拟合模型稳定性为{:.2%}".format(np.abs((y0_exp2[1]-y0_exp1[1])/y0_exp1[1])))

plt.show()
#####最开始可以引入指数增长模型，但介绍中国人口受政策（计划生育等）影响较大，导致其人数在时间序列上呈现线性，即使是使用线性模型也有不错的效果，










plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
#1次多项式拟合
c = np.polyfit(x_1, y_1ln, 1) #用1次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_1 = f(xi_3)
#yi_power_nihe_2=np.exp(yi_ln_nihe_1)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_ln_nihe_1 , label='一次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数一次函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,2)
#2次多项式拟合
c = np.polyfit(x_1, y_1ln, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_2 = f(xi_3)
#yi_power_nihe_2=np.exp(yi_ln_nihe_2)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_ln_nihe_2 , label='二次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数二次函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,3)
#3次多项式拟合
c = np.polyfit(x_1, y_1ln, 3) #用3次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_3 = f(xi_3)
#yi_power_nihe_2=np.exp(yi_ln_nihe_3)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_ln_nihe_3 , label='三次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数三次函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,4)
#4次多项式拟合
c = np.polyfit(x_1, y_1ln, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_4 = f(xi_3)
#yi_power_nihe_2=np.exp(yi_ln_nihe_4)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_ln_nihe_4 , label='四次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数四次函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,5)
#5次多项式拟合
c = np.polyfit(x_1, y_1ln, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_5 = f(xi_3)
#yi_power_nihe_2=np.exp(yi_ln_nihe_5)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_ln_nihe_5 , label='五次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数五次函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,6)
#6次多项式拟合
c = np.polyfit(x_1, y_1ln, 6) #用6次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_6 = f(xi_3)
#yi_power_nihe_6=np.exp(yi_ln_nihe_6)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_ln_nihe_6 , label='六次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数六次函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.show()


plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
#1次多项式拟合
c = np.polyfit(x_1, y_1ln, 1) #用3次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_1 = f(xi_3)
yi_power_nihe_1=np.exp(yi_ln_nihe_1)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_power_nihe_1 , label='一次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数一次幂函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,2)
#2次多项式拟合
c = np.polyfit(x_1, y_1ln, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_2 = f(xi_3)
yi_power_nihe_2=np.exp(yi_ln_nihe_2)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_power_nihe_2 , label='二次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数二次幂函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,3)
#3次多项式拟合
c = np.polyfit(x_1, y_1ln, 3) #用3次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_3 = f(xi_3)
yi_power_nihe_3=np.exp(yi_ln_nihe_3)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_power_nihe_3 , label='三次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数三次幂函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,4)
#4次多项式拟合
c = np.polyfit(x_1, y_1ln, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_4 = f(xi_3)
yi_power_nihe_4=np.exp(yi_ln_nihe_4)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_power_nihe_4 , label='四次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数四次幂函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,5)
#5次多项式拟合
c = np.polyfit(x_1, y_1ln, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_5 = f(xi_3)
yi_power_nihe_5=np.exp(yi_ln_nihe_5)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_power_nihe_5 , label='五次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数五次幂函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,6)
#6次多项式拟合
c = np.polyfit(x_1, y_1ln, 6) #用6次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_ln_nihe_6 = f(xi_3)
yi_power_nihe_6=np.exp(yi_ln_nihe_6)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_power_nihe_6 , label='六次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口六次幂函数拟合",fontdict={"fontsize":14})
plt.legend()
plt.show()


#########################################稳定性分析


plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
#1次多项式拟合
c = np.polyfit(x_1, y_1ln, 1) #用1次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_1 = f(xi_3)
#1次多项式拟合
c = np.polyfit(x_2, y_2ln, 1) #用1次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_1 = f(xi_4)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_1ln_nihe_1 , label='1949-2010一次函数拟合')
plt.plot(xi_4, yi_2ln_nihe_1 , label='1949-2020一次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数一次函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,2)
#2次多项式拟合
c = np.polyfit(x_1, y_1ln, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_2 = f(xi_3)
#2次多项式拟合
c = np.polyfit(x_2, y_2ln, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_2 = f(xi_4)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_1ln_nihe_2 , label='1949-2010二次函数拟合')
plt.plot(xi_4, yi_2ln_nihe_2 , label='1949-2020二次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数二次函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,3)
#3次多项式拟合
c = np.polyfit(x_1, y_1ln, 3) #用3次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_3 = f(xi_3)
#3次多项式拟合
c = np.polyfit(x_2, y_2ln, 3) #用3次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_3 = f(xi_4)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_1ln_nihe_3 , label='1949-2010三次函数拟合')
plt.plot(xi_4, yi_2ln_nihe_3 , label='1949-2020三次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数三次函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,4)
#4次多项式拟合
c = np.polyfit(x_1, y_1ln, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_4 = f(xi_3)
#4次多项式拟合
c = np.polyfit(x_2, y_2ln, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_4 = f(xi_4)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_1ln_nihe_4 , label='1949-2010四次函数拟合')
plt.plot(xi_4, yi_2ln_nihe_4 , label='1949-2020四次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数四次函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,5)
#5次多项式拟合
c = np.polyfit(x_1, y_1ln, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_5 = f(xi_3)
#5次多项式拟合
c = np.polyfit(x_2, y_2ln, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_5 = f(xi_4)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_1ln_nihe_5 , label='1949-2010五次函数拟合')
plt.plot(xi_4, yi_2ln_nihe_5 , label='1949-2020五次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数五次函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,6)
#6次多项式拟合
c = np.polyfit(x_1, y_1ln, 6) #用6次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_6 = f(xi_3)
#6次多项式拟合
c = np.polyfit(x_2, y_2ln, 6) #用6次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_6 = f(xi_4)
plt.scatter(2020, np.log(14.43), label='待预测函数节点的真值ln')
plt.scatter(x_1, y_1ln, label='待拟合函数节点ln')
plt.plot(xi_3, yi_1ln_nihe_6 , label='1949-2010五次函数拟合')
plt.plot(xi_4, yi_2ln_nihe_6 , label='1949-2020五次函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口取对数六次函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.show()




















plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
#1次多项式拟合
c = np.polyfit(x_1, y_1ln, 1) #用1次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_1 = f(xi_3)
yi_1power_nihe_1=np.exp(yi_1ln_nihe_1)
y0_1=np.exp(f(x0))
print("1次多项式指数拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_1[0],y0_1[1]))
#1次多项式拟合
c = np.polyfit(x_2, y_2ln, 1) #用1次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_1 = f(xi_4)
yi_2power_nihe_1=np.exp(yi_2ln_nihe_1)
y1_1=np.exp(f(x0))
print("1次多项式指数拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_1[1]))
print("1次多项式指数拟合模型误差为{:.2%}".format(np.abs((y0_1[0]-y_2[-1])/y_2[-1])))
print("1次多项式指数拟合模型稳定性为{:.2%}".format(np.abs((y1_1[1]-y0_1[1])/y0_1[1])))

plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_1power_nihe_1 , label='1949-2010一次幂函数拟合')
plt.plot(xi_4, yi_2power_nihe_1 , label='1949-2020一次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口一次幂函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,2)
#2次多项式拟合
c = np.polyfit(x_1, y_1ln, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_2 = f(xi_3)
yi_1power_nihe_2=np.exp(yi_1ln_nihe_2)
y0_2=np.exp(f(x0))
print("2次多项式指数拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_2[0],y0_2[1]))
#2次多项式拟合
c = np.polyfit(x_2, y_2ln, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_2 = f(xi_4)
yi_2power_nihe_2=np.exp(yi_2ln_nihe_2)
y1_2=np.exp(f(x0))
print("2次多项式指数拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_2[1]))
print("2次多项式指数拟合模型误差为{:.2%}".format(np.abs((y0_2[0]-y_2[-1])/y_2[-1])))
print("2次多项式指数拟合模型稳定性为{:.2%}".format(np.abs((y1_2[1]-y0_2[1])/y0_2[1])))

plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_1power_nihe_2 , label='1949-2010二次幂函数拟合')
plt.plot(xi_4, yi_2power_nihe_2 , label='1949-2020二次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口二次幂函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,3)
#3次多项式拟合
c = np.polyfit(x_1, y_1ln, 3) #用3次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_3 = f(xi_3)
yi_1power_nihe_3=np.exp(yi_1ln_nihe_3)
y0_3=np.exp(f(x0))
print("3次多项式指数拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_3[0],y0_3[1]))
#3次多项式拟合
c = np.polyfit(x_2, y_2ln, 3) #用3次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_3 = f(xi_4)
yi_2power_nihe_3=np.exp(yi_2ln_nihe_3)
y1_3=np.exp(f(x0))
print("3次多项式指数拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_3[1]))
print("3次多项式指数拟合模型误差为{:.2%}".format(np.abs((y0_3[0]-y_2[-1])/y_2[-1])))
print("3次多项式指数拟合模型稳定性为{:.2%}".format(np.abs((y1_3[1]-y0_3[1])/y0_3[1])))

plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_1power_nihe_3 , label='1949-2010三次幂函数拟合')
plt.plot(xi_4, yi_2power_nihe_3 , label='1949-2020三次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口三次幂函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,4)
#4次多项式拟合
c = np.polyfit(x_1, y_1ln, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_4 = f(xi_3)
yi_1power_nihe_4=np.exp(yi_1ln_nihe_4)
y0_4=np.exp(f(x0))
print("4次多项式指数拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_4[0],y0_4[1]))
#4次多项式拟合
c = np.polyfit(x_2, y_2ln, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_4 = f(xi_4)
yi_2power_nihe_4=np.exp(yi_2ln_nihe_4)
y1_4=np.exp(f(x0))
print("4次多项式指数拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_4[1]))
print("4次多项式指数拟合模型误差为{:.2%}".format(np.abs((y0_4[0]-y_2[-1])/y_2[-1])))
print("4次多项式指数拟合模型稳定性为{:.2%}".format(np.abs((y1_4[1]-y0_4[1])/y0_4[1])))

plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_1power_nihe_4 , label='1949-2010四次幂函数拟合')
plt.plot(xi_4, yi_2power_nihe_4, label='1949-2020四次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口四次幂函数拟合对",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,5)
#5次多项式拟合
c = np.polyfit(x_1, y_1ln, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_5 = f(xi_3)
yi_1power_nihe_5=np.exp(yi_1ln_nihe_5)
y0_5=np.exp(f(x0))
print("5次多项式指数拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_5[0],y0_5[1]))
#5次多项式拟合
c = np.polyfit(x_2, y_2ln, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_5 = f(xi_4)
yi_2power_nihe_5=np.exp(yi_2ln_nihe_5)
y1_5=np.exp(f(x0))
print("5次多项式指数拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_5[1]))
print("5次多项式指数拟合模型误差为{:.2%}".format(np.abs((y0_5[0]-y_2[-1])/y_2[-1])))
print("5次多项式指数拟合模型稳定性为{:.2%}".format(np.abs((y1_5[1]-y0_5[1])/y0_5[1])))

plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_1power_nihe_5 , label='1949-2010五次幂函数拟合')
plt.plot(xi_4, yi_2power_nihe_5 , label='1949-2020五次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口五次幂函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.subplot(2,3,6)
#6次多项式拟合
c = np.polyfit(x_1, y_1ln, 6) #用6次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_1ln_nihe_6 = f(xi_3)
yi_1power_nihe_6=np.exp(yi_1ln_nihe_6)
y0_6=np.exp(f(x0))
print("6次多项式指数拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_6[0],y0_6[1]))
#6次多项式拟合
c = np.polyfit(x_2, y_2ln, 6) #用6次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_6 = f(xi_4)
yi_2power_nihe_6=np.exp(yi_2ln_nihe_6)
y1_6=np.exp(f(x0))
print("6次多项式指数拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_6[1]))
print("6次多项式指数拟合模型误差为{:.2%}".format(np.abs((y0_6[0]-y_2[-1])/y_2[-1])))
print("6次多项式指数拟合模型稳定性为{:.2%}".format(np.abs((y1_6[1]-y0_6[1])/y0_6[1])))

plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_1power_nihe_6 , label='1949-2010五次幂函数拟合')
plt.plot(xi_4, yi_2power_nihe_6 , label='1949-2020五次幂函数拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.title("人口六次幂函数拟合对比",fontdict={"fontsize":14})
plt.legend()
plt.show()




























#画图
plt.figure(figsize=(8,4))
c = np.polyfit(x_2, y_2ln, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_2ln_nihe_2 = f(xi_4)
yi_power_nihe_2=np.exp(yi_2ln_nihe_2)
#plt.scatter(x_1, y_1, label='待拟合函数节点')
#plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.scatter(x_2, y_2,c='g', label='待拟合函数节点')
plt.plot(xi_4, yi_power_nihe_2 , label='二次多项式拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
#plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("指数模型的最小二乘拟合半对数坐标",fontdict={"fontsize":14})
plt.yscale("log")
plt.legend()
plt.show()