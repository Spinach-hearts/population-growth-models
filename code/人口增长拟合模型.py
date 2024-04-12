# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:12:04 2023

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
xi_1=np.linspace(1949,2020,100)
xi_2=np.linspace(1949,2030,100)
xi_3=range(1949,2023,1)
xi_4=range(1949,2033,1)
x0=np.array([2020,2030])
#拟合节点共计7个，故使用多项式函数拟合的最高次数只有6次，故将1-6次的拟合结果以图象显示，
#画图
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
#1次多项式拟合
c = np.polyfit(x_1, y_1, 1) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_1 = f(xi_3)
y0_1=f(x0)
print("1次多项式拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_1[0],y0_1[1]))
#2次多项式拟合
c = np.polyfit(x_1, y_1, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_2 = f(xi_3)
y0_2=f(x0)
print("2次多项式拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_2[0],y0_2[1]))
#3次多项式拟合
c = np.polyfit(x_1, y_1, 3) #用3次多项式拟合，
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_3 = f(xi_3)
y0_3=f(x0)
print("3次多项式拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_3[0],y0_3[1]))
#4次多项式拟合
c = np.polyfit(x_1, y_1, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_4 = f(xi_3)
y0_4=f(x0)
print("4次多项式拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_4[0],y0_4[1]))
#5次多项式拟合
c = np.polyfit(x_1, y_1, 5) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_5 = f(xi_3)
y0_5=f(x0)
print("5次多项式拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_5[0],y0_5[1]))
#6次多项式拟合
c = np.polyfit(x_1, y_1, 6) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_6 = f(xi_3)
y0_6=f(x0)
print("6次多项式拟合基于1949-2010年数据预测2020年人口数量为{:.2f}亿/人，2030年人口数量为{:.2f}亿/人".format(y0_6[0],y0_6[1]))
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_nihe_1, label='一次多项式拟合')
plt.plot(xi_3, yi_nihe_2, label='两次多项式拟合')
plt.plot(xi_3, yi_nihe_3, label='三次多项式拟合')
plt.plot(xi_3, yi_nihe_4, label='四次多项式拟合')
plt.plot(xi_3, yi_nihe_5, label='五次多项式拟合')
plt.plot(xi_3, yi_nihe_6, label='六次多项式拟合')
plt.xlabel("x（年份）",fontdict={"fontsize":12})
plt.ylabel("y（亿）",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("1949-2010年的人口增长多项式拟合模型",fontdict={"fontsize":14})
plt.legend()

plt.subplot(2,2,2)
#1次多项式拟合
c = np.polyfit(x_2, y_2, 1) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_1 = f(xi_4)
y1_1=f(x0)
print("1次多项式拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_1[1]))
#2次多项式拟合
c = np.polyfit(x_2, y_2, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_2 = f(xi_4)
y1_2=f(x0)
print("2次多项式拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_2[1]))
#3次多项式拟合
c = np.polyfit(x_2, y_2, 3) #用3次多项式拟合，
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_3 = f(xi_4)
y1_3=f(x0)
print("3次多项式拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_3[1]))
#4次多项式拟合
c = np.polyfit(x_2, y_2, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_4 = f(xi_4)
y1_4=f(x0)
print("4次多项式拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_4[1]))
#5次多项式拟合
c = np.polyfit(x_2, y_2, 5) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_5 = f(xi_4)
y1_5=f(x0)
print("5次多项式拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_5[1]))
#6次多项式拟合
c = np.polyfit(x_2, y_2, 6) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe_6 = f(xi_4)
y1_6=f(x0)
print("6次多项式拟合基于1949-2020年数据预测2030年人口数量为{:.2f}亿/人".format(y1_6[1]))
plt.scatter(x_2, y_2, label='待拟合函数节点')
plt.plot(xi_4, yi_nihe_1, label='一次多项式拟合')
plt.plot(xi_4, yi_nihe_2, label='两次多项式拟合')
plt.plot(xi_4, yi_nihe_3, label='三次多项式拟合')
plt.plot(xi_4, yi_nihe_4, label='四次多项式拟合')
plt.plot(xi_4, yi_nihe_5, label='五次多项式拟合')
plt.plot(xi_4, yi_nihe_6, label='六次多项式拟合')
plt.xlabel("x（年份）",fontdict={"fontsize":12})
plt.ylabel("y（亿）",fontdict={"fontsize":12})
plt.vlines(2030,6,24,'g','dashed',label='2030年')
plt.title("1949-2020年的人口增长多项式拟合模型",fontdict={"fontsize":14})
plt.legend()

print("1次多项式拟合模型误差为{:.2%}".format(np.abs((y0_1[0]-y_2[-1])/y_2[-1])))
print("1次多项式拟合模型稳定性为{:.2%}".format(np.abs((y1_1[1]-y0_1[1])/y0_1[1])))
print("2次多项式拟合模型误差为{:.2%}".format(np.abs((y0_2[0]-y_2[-1])/y_2[-1])))
print("2次多项式拟合模型稳定性为{:.2%}".format(np.abs((y1_2[1]-y0_2[1])/y0_2[1])))

print("3次多项式拟合模型误差为{:.2%}".format(np.abs((y0_3[0]-y_2[-1])/y_2[-1])))
print("3次多项式拟合模型稳定性为{:.2%}".format(np.abs((y1_3[1]-y0_3[1])/y0_3[1])))
print("4次多项式拟合模型误差为{:.2%}".format(np.abs((y0_4[0]-y_2[-1])/y_2[-1])))
print("4次多项式拟合模型稳定性为{:.2%}".format(np.abs((y1_4[1]-y0_4[1])/y0_4[1])))

print("5次多项式拟合模型误差为{:.2%}".format(np.abs((y0_5[0]-y_2[-1])/y_2[-1])))
print("5次多项式拟合模型稳定性为{:.2%}".format(np.abs((y1_5[1]-y0_5[1])/y0_5[1])))
print("6次多项式拟合模型误差为{:.2%}".format(np.abs((y0_6[0]-y_2[-1])/y_2[-1])))
print("6次多项式拟合模型稳定性为{:.2%}".format(np.abs((y1_6[1]-y0_6[1])/y0_6[1])))


plt.show()
#拟合函数图像重合过多，不易分析，故而将其分组，这里依次数，两个一组，


########################################稳定性，准确性一起分析，每一组，寻找最优的一个。
#画图  函数拟合模型
plt.figure(figsize=(16,8))
plt.subplot(3,1,1)
#1次多项式拟合
c = np.polyfit(x_1, y_1, 1) #用1次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_1 = f(xi_3)
#2次多项式拟合
c = np.polyfit(x_1, y_1, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_2 = f(xi_3)
#############################################################
#1次多项式拟合
c = np.polyfit(x_2, y_2, 1) #用1次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_1 = f(xi_4)
#2次多项式拟合
c = np.polyfit(x_2, y_2, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_2 = f(xi_4)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_nihe1_1, label='1949-2010一次多项式拟合')
plt.plot(xi_3, yi_nihe1_2, label='1949-2010两次多项式拟合')
#######################################################
plt.plot(xi_4, yi_nihe2_1, label='1949-2020一次多项式拟合')
plt.plot(xi_4, yi_nihe2_2, label='1949-2020两次多项式拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("函数拟合模型",fontdict={"fontsize":14})
plt.legend()
plt.subplot(3,1,2)
#3次多项式拟合
c = np.polyfit(x_1, y_1, 3) #用3次多项式拟合，
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_3 = f(xi_3)
#4次多项式拟合
c = np.polyfit(x_1, y_1, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_4 = f(xi_3)
#############################################################
#3次多项式拟合
c = np.polyfit(x_2, y_2, 3) #用3次多项式拟合，
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_3 = f(xi_4)
#4次多项式拟合
c = np.polyfit(x_2, y_2, 4) #用4次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_4 = f(xi_4)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_nihe1_3, label='1949-2010三次多项式拟合')
plt.plot(xi_3, yi_nihe1_4, label='1949-2010四次多项式拟合')
#######################################################
plt.plot(xi_4, yi_nihe2_3, label='1949-2020三次多项式拟合')
plt.plot(xi_4, yi_nihe2_4, label='1949-2020四次多项式拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.legend()
plt.subplot(3,1,3)
#5次多项式拟合
c = np.polyfit(x_1, y_1, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_5 = f(xi_3)
#6次多项式拟合
c = np.polyfit(x_1, y_1, 6) #用6次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_6 = f(xi_3)
####################################################################
#5次多项式拟合
c = np.polyfit(x_2, y_2, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_5 = f(xi_4)
#6次多项式拟合
c = np.polyfit(x_2, y_2, 6) #用6次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_6 = f(xi_4)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_nihe1_5, label='1949-2010五次多项式拟合')
plt.plot(xi_3, yi_nihe1_6, label='1949-2010六次多项式拟合')
#############################################################
plt.plot(xi_4, yi_nihe2_5, label='1949-2020五次多项式拟合')
plt.plot(xi_4, yi_nihe2_6, label='1949-2020六次多项式拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.legend()
plt.show()


##由拟合函数性质知，7个拟合节点，至多可以构造6次多项式，故而6次之外的拟合函数不必讨论，其中四次、六次的多项式拟合的稳定性不好且加入新数据后，预测结果有较大误差，故而不再考虑，而一次多项式拟合结果的准确性与稳定性皆弱于二次函数，故不讨论。


#####################################2，3，5次模型
#画图  
plt.figure(figsize=(8,4))
#2次多项式拟合
c = np.polyfit(x_1, y_1, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_2 = f(xi_3)
#2次多项式拟合
c = np.polyfit(x_2, y_2, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_2 = f(xi_4)
#3次多项式拟合
c = np.polyfit(x_1, y_1, 3) #用3次多项式拟合，
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_3 = f(xi_3)
#3次多项式拟合
c = np.polyfit(x_2, y_2, 3) #用3次多项式拟合，
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_3 = f(xi_4)
#5次多项式拟合
c = np.polyfit(x_1, y_1, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_5 = f(xi_3)
#5次多项式拟合
c = np.polyfit(x_2, y_2, 5) #用5次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_5 = f(xi_4)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_nihe1_2, label='1949-2010两次多项式拟合')
plt.plot(xi_4, yi_nihe2_2, label='1949-2020两次多项式拟合')
plt.plot(xi_3, yi_nihe1_3, label='1949-2010三次多项式拟合')
plt.plot(xi_4, yi_nihe2_3, label='1949-2020三次多项式拟合')
plt.plot(xi_3, yi_nihe1_5, label='1949-2010五次多项式拟合')
plt.plot(xi_4, yi_nihe2_5, label='1949-2020五次多项式拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("2、3、5次多项式拟合函数模型",fontdict={"fontsize":14})
plt.legend()
plt.show()

#基于1949-2010数据，两次模型与五次模型预测较为准确，且两者差值较小，然三次误差较大；基于1949-2020数据，五次方程给出了较大的预测结果，这显然在当今人口年龄分布下，不会发生，故该模型不再考虑。三次与两次拟合函数的预测较为接近，但两次拟合函数的稳定性要明显优于三次，其可信度更高。但三次预估会下降，更符合我对中国人口的认识，故也保留。
#####################################2，3次模型
#画图  
plt.figure(figsize=(8,4))
#2次多项式拟合
c = np.polyfit(x_1, y_1, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_2 = f(xi_3)
#2次多项式拟合
c = np.polyfit(x_2, y_2, 2) #用2次多项式拟合，输出系数从高到0
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_2 = f(xi_4)
#3次多项式拟合
c = np.polyfit(x_1, y_1, 3) #用3次多项式拟合，
f = np.poly1d(c) #使用次数合成多项式
yi_nihe1_3 = f(xi_3)
#3次多项式拟合
c = np.polyfit(x_2, y_2, 3) #用3次多项式拟合，
f = np.poly1d(c) #使用次数合成多项式
yi_nihe2_3 = f(xi_4)
plt.scatter(2020, 14.43, label='待预测函数节点的真值')
plt.scatter(x_1, y_1, label='待拟合函数节点')
plt.plot(xi_3, yi_nihe1_2, label='1949-2010两次多项式拟合')
plt.plot(xi_4, yi_nihe2_2, label='1949-2020两次多项式拟合')
plt.plot(xi_3, yi_nihe1_3, label='1949-2010三次多项式拟合')
plt.plot(xi_4, yi_nihe2_3, label='1949-2020三次多项式拟合')
plt.xlabel("x",fontdict={"fontsize":12})
plt.ylabel("y",fontdict={"fontsize":12})
plt.vlines(2020,6,20,'g','dashed',label='2020年')
plt.title("2、3次多项式拟合函数模型",fontdict={"fontsize":14})
plt.legend()
plt.show()




#评估模型的好坏
#准确度：函数在2020年的预测值与真值的比较，可以定量
#稳定性：增加节点，函数的在2020后的变化，可以定性
#经验：期待的模型是2020年后递减的部分，至少增加的很缓慢。可以定性。


