# 梯度下降（Gradient Descent）

在求解机器学习算法的模型参数，即无约束优化问题时，梯度下降（Gradient Descent）是最常采用的方法之一，另一种常用的方法是最小二乘法。这里就对梯度下降法做一个完整的总结。

## 1. 梯度
在微积分里面，对多元函数的参数求∂偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度。比如函数f(x,y), 分别对x,y求偏导数，求得的梯度向量就是$$(\partial f/\partial x, \partial f/\partial y)^T$$,简称grad f(x,y)或者▽f(x,y)。对于在点$$(x_0,y_0)$$的具体梯度向量就是$$(\partial f/\partial x0,\partial f/\partial y0)^T$$.或者▽f(x_0,y_0)，如果是3个参数的向量梯度，就是$$(\partial f/\partial x, \partial f/\partial y，\partial f/\partial z)^T$$,以此类推。

