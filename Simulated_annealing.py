#!/usr/bin/env
# coding:utf-8

__author__ = 'DJR'

'''模拟退火算法：
一种贪心算法，但是它的搜索过程引入随机因素。模拟退火算法以一定的概率来接受一个比当前解要差的解，因此有可能会跳出这个局部的最优解，达到全局的最优解
1.随机算法，不一定找到全局的最优解；
2.搜索效率比枚举法高，常用于解决快速求解近似最优解问题。

求解过程：
1.设置参数
2.初始解（随机产生一个初始路径）
3.Metropolis准则:设置路径函数为 fit(S)，则 S_1, S_2 的路径差 df = fit(S_2) - fit(S_1)
   以概率P来选择是否接受新的路径：如果 df<0，则接受新的路径 S_2，否则以概率 exp(-df/T)接受新的路径
4.降温 ：利用降温速率 r 进行降温， T=r*T，直至 T<T_end停止迭代。

问题描述：
1.旅行商问题（TSP）
'''

from math import exp
import numpy as np
import matplotlib.pyplot as plt


class SA_TSP():
    def __init__(self, data, T=1000, T_end=1e-3, L=200, r=0.9):
        self.T = T  # 初始温度
        self.T_end = T_end  # 终止温度
        self.data = data  # 位置坐标
        self.num = len(data)  # 城市个数
        self.L = L  # 每一个温度下的链长
        self.r = r  # 降温速率
        self.matrix_distance = self.matrix_dis()  # 距离矩阵
        self.chrom = np.array([0] * self.num)  # 初始化路径和距离
        self.fitness = 0
        self.new_chrom = np.array([0] * self.num)  # 变换后的路径和距离
        self.new_fitness = 0

    # 计算两个城市间的距离
    def matrix_dis(self):
        res = np.zeros((self.num, self.num))
        for i in range(self.num):
            for j in range(i+1, self.num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])
                res[j, i] = res[i, j]
        return res

    # 初始化路径
    def rand_chrom(self):
        rand_ch = np.array(range(self.num))
        np.random.shuffle(rand_ch)
        self.chrom = rand_ch
        self.fitness = self.comp_fit(rand_ch)

    # 计算一个路径距离值，可利用该函数更新self.fittness
    def comp_fit(self, one_path):
        res = 0
        for i in range(self.num - 1):
            res += self.matrix_distance[one_path[i], one_path[i+1]]
        res += self.matrix_distance[one_path[-1], one_path[0]]
        return res

    # 路径可视化
    def out_path(self, one_path):
        res = str(one_path[0] + 1) + '-->'
        for i in range(1, self.num):
            res += str(one_path[i] + 1) + '-->'
        res += str(one_path[0] + 1) + '\n'
        print(res)

    # 更新交换后的路径和距离
    def new_way_1(self):
        self.new_chrom = self.chrom.copy()
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)
        self.new_chrom[r1], self.new_chrom[r2] = self.new_chrom[r2], self.new_chrom[r1]
        self.new_fitness = self.comp_fit(self.new_chrom)

    # Metropolis准则
    def metropolis(self):
        df = self.new_fitness - self.fitness
        # 新路径更短 接受为新解
        if df < 0:
            self.chrom = self.new_chrom.copy()
            self.fitness = self.new_fitness
        else:
            if exp(-df/self.T) > np.random.rand():
                self.chrom = self.new_chrom.copy()
                self.fitness = self.new_fitness


# 旅行商问题
def TSP_main(data):
    Path_short = SA_TSP(data, T=5000, r=0.9, L=400)
    Path_short.rand_chrom()  # 初始化路径
    # 绘制初始化路径
    fig = plt.figure()  # 生成画布
    plt.ion()  # 打开交互模式
    print('旅行商的初始路程: ')
    Path_short.out_path(Path_short.chrom)
    print('距离: ' + str(Path_short.fitness))

    # 存储退火过程中的最优路径变化
    Path_short.best_chrom = [Path_short.chrom]
    Path_short.best_fit = [Path_short.fitness]

    while Path_short.T > Path_short.T_end:
        fig.clf()  # 清空当前Figure对象
        chrom, fit = [], []  # 存储每一个退火过程的路径和距离找寻最优
        for i in range(Path_short.L):
            Path_short.new_way_1()  # 变换产生新路径
            Path_short.metropolis()  # 判断是否接受新路径

            chrom.append(Path_short.chrom)
            fit.append(Path_short.fitness)

        # 存储该步迭代后的最优路径
        index = np.argmin(fit)
        if fit[index] >= Path_short.best_fit[-1]:
            Path_short.best_fit.append(Path_short.best_fit[-1])
            Path_short.best_chrom.append(Path_short.best_chrom[-1])
        else:
            Path_short.best_chrom.append(chrom[index])
            Path_short.best_fit.append(fit[index])

        # 更新温度
        Path_short.T *= Path_short.r
        # 绘制图片
        ax = fig.add_subplot()
        x, y = data[:, 0], data[:, 1]
        ax.scatter(x, y, color='g', linewidths=0.1)
        for i, txt in enumerate(range(1, len(data) + 1)):
            ax.annotate(txt, (x[i], y[i]))
        res_0 = Path_short.chrom
        x_0, y_0 = x[res_0], y[res_0]
        for i in range(len(data) - 1):
            plt.quiver(x_0[i], y_0[i], x_0[i + 1] - x_0[i], y_0[i + 1] - y_0[i], color='r', width=0.005, angles='xy',
                       scale=1,
                       scale_units='xy')
        plt.quiver(x_0[-1], y_0[-1], x_0[0] - x_0[-1], y_0[0] - y_0[-1], color='r', width=0.005, angles='xy', scale=1,
                   scale_units='xy')
        plt.pause(0.1)  # 暂停

    # 关闭交互模式
    plt.ioff()
    plt.show()
    print('旅行商的最终路程: ')
    Path_short.out_path(Path_short.chrom)
    print('距离：' + str(Path_short.fitness))
    return Path_short


if __name__ == '__main__':
    print('模拟退火案例--旅行商问题')
    # 随机生成30个城市坐标
    np.random.seed(10)
    data = np.random.rand(30, 2) * 10
    TSP_main(data)


