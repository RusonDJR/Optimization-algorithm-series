# Optimization-algorithm-series
Simulated Annealing（模拟退火），Genetic Algorithm（遗传算法）


1. 模拟退火算法
模拟退火算法是一种贪心算法，但是它的搜索过程引入随机因素。模拟退火算法以一定的概率来接受一个比当前解要差的解，因此有可能会跳出这个局部的最优解，达到全局的最优解

1.随机算法，不一定找到全局的最优解；

2.搜索效率比枚举法高，常用于解决快速求解近似最优解问题。

求解过程：

1.设置参数

2.初始解（随机产生一个初始路径）

3.Metropolis准则:设置路径函数为 fit(S)，则 S_1, S_2 的路径差 df = fit(S_2) - fit(S_1)
   以概率P来选择是否接受新的路径：如果 df<0，则接受新的路径 S_2，否则以概率 exp(-df/T)接受新的路径
   
4.降温 ：利用降温速率 r 进行降温， T=r*T，直至 T<T_end停止迭代。

算法应用：

1.Travelling salesman problem（TSP，旅行推销员问题）


