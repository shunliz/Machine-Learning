# **牛顿法介绍**

**牛顿法**也是求解**无约束最优化**问题常用的方法，**最大的优点是收敛速度快**。

从本质上去看，**牛顿法是二阶收敛，梯度下降是一阶收敛，所以牛顿法就更快**。**通俗地说**，比如你想找一条最短的路径走到一个盆地的最底部，梯度下降法 每次只从你当前所处位置选一个坡度最大的方向走一步，牛顿法在选择方向时，不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大。所以， 可以说牛顿法比梯度下降法看得更远一点，能更快地走到最底部。

![](/assets/newton1.png)

或者从几何上说，**牛顿法就是用一个二次曲面去拟合你当前所处位置的局部曲面，而梯度下降法是用一个平面去拟合当前的局部曲面**，通常情况下，二次曲面的拟合会比平面更好，所以牛顿法选择的下降路径会更符合真实的最优下降路径。

## **2、牛顿法的推导**

将目标函数f\(x\)在$$x_k$$处进行二阶泰勒展开，可得：

$$f(x)=f\left(x_{k}\right)+f^{\prime}\left(x_{k}\right)\left(x-x_{k}\right)+\frac{1}{2} f^{\prime \prime}\left(x_{k}\right)\left(x-x_{k}\right)^{2}$$

因为目标函数f\(x\)有极值的必要条件是在极值点处一阶导数为0，即：$$f^{\prime}(x)=0$$

所以对上面的展开式两边同时求导（注意x才是变量，$$x_k$$是常量$$\Rightarrow f^{\prime}\left(x_{k}\right), f^{\prime \prime}\left(x_{k}\right)$$都是常量），并令$$f^{\prime}(x)=0$$可得：

$$f^{\prime}\left(x_{k}\right)+f^{\prime \prime}\left(x_{k}\right)\left(x-x_{k}\right)=0$$

即：

$$x=x_{k}-\frac{f^{\prime}\left(x_{k}\right)}{f^{\prime \prime}\left(x_{k}\right)}$$

于是可以构造如下的迭代公式：

$$x_{k+1}=x_{k}-\frac{f^{\prime}\left(x_{k}\right)}{f^{\prime \prime}\left(x_{k}\right)}$$

这样，我们就可以利用该迭代式依次产生的序列$$\left\{x_{1}, x_{2}, \ldots, x_{k}\right\}$$才逐渐逼近f\(x\)的极小值点了。

**高维情况**的牛顿迭代公式是：

$$\mathbf{x}_{n+1}=\mathbf{x}_{n}-\left[H f\left(\mathbf{x}_{n}\right)\right]^{-1} \nabla f\left(\mathbf{x}_{n}\right), n \geq 0$$

式中， ▽f是f\(x\)的梯度，即：

$$\nabla f=\left[ \begin{array}{c}{\frac{\partial f}{\partial x_{1}}} \\ {\frac{\partial f}{\partial x_{2}}} \\ {\vdots} \\ {\frac{\partial f}{\partial x_{N}}}\end{array}\right]$$

H是Hessen矩阵，即：

$$H(f)=\left[ \begin{array}{cccc}{\frac{\partial^{2} f}{\partial x_{1}^{2}}} & {\frac{\partial^{2} f}{\partial x_{1} \partial x_{2}}} & {\cdots} & {\frac{\partial^{2} f}{\partial x_{1} \partial x_{n}}} \\ {\frac{\partial^{2} f}{\partial x_{2} \partial x_{1}}} & {\frac{\partial^{2} f}{\partial x_{2}^{2}}} & {\cdots} & {\frac{\partial^{2} f}{\partial x_{2} \partial x_{n}}} \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial^{2} f}{\partial x_{n} \partial x_{1}}} & {\frac{\partial^{2} f}{\partial x_{n} \partial x_{2}}} & {\cdots} & {\frac{\partial^{2} f}{\partial x_{n}^{2}}}\end{array}\right]$$

## **3、牛顿法的过程**

* 1、给定初值$$x_0$$和精度阈值$$\varepsilon$$，并令$$k=0$$；
* 2、计算$$x_k$$和$$H_K$$；
* 3、若$$\left\|g_{k}\right\|<\varepsilon$$ 则停止迭代；否则确定搜索方向：$$d_{k}=-H_{k}^{-1} \cdot g_{k}$$；
* 4、计算新的迭代点：$$x_{k+1}=x_{k}+d_{k}$$；
* 5、令k=k+1，转至2。

## 4、**阻尼牛顿法**

### **1、引入**

注意到，牛顿法的迭代公式中没有步长因子，是定步长迭代。对于非二次型目标函数，有时候会出现$$f\left(x_{k+1}\right)>f\left(x_{k}\right)$$的情况，这表明，原始牛顿法不能保证函数值稳定的下降。在严重的情况下甚至会造成序列发散而导致计算失败。

为消除这一弊病，人们又提出阻尼牛顿法。阻尼牛顿法每次迭代的方向仍然是$$x_k$$，但每次迭代会沿此方向做一维搜索，寻求最优的步长因子$$\lambda_k$$，即：$$\lambda_{k}=\min f\left(x_{k}+\lambda d_{k}\right)$$

### **2、算法过程**

* 1、给定初值$$x = y$$和精度阈值$$\varepsilon$$，并令k=0；
* 2、计算$$g_k$$（f\(x\)在$$x_k$$处的梯度值）和$$H_k$$；
* 3、若$$\left\|g_{k}\right\|<\varepsilon$$ 则停止迭代；否则确定搜索方向：$$d_{k}=-H_{k}^{-1} \cdot g_{k}$$；
* 4、利用$$d_{k}=-H_{k}^{-1} \cdot g_{k}$$得到步长$$\lambda_k$$，并令
  $$x_{k+1}=x_{k}+\lambda_{k} d_{k}$$
* 5、令k=k+1，转至2。

## 5、**拟牛顿法**

### **1、概述**

由于**牛顿法**每一步都要求解目标函数的**Hessen矩阵的逆矩阵**，**计算量比较大**（求矩阵的逆运算量比较大），因此提出一种**改进方法**，即**通过正定矩阵近似代替Hessen矩阵的逆矩阵，简化这一计算过程**，改进后的方法称为**拟牛顿法**。

### **2、拟牛顿法的推导**

先将目标函数在$$x_{k+1}$$处展开，得到：

$$f(x)=f\left(x_{k+1}\right)+f^{\prime}\left(x_{k+1}\right)\left(x-x_{k+1}\right)+\frac{1}{2} f^{\prime \prime}\left(x_{k+1}\right)\left(x-x_{k+1}\right)^{2}$$

两边同时取梯度，得：

$$f^{\prime}(x)=f^{\prime}\left(x_{k+1}\right)+f^{\prime \prime}\left(x_{k+1}\right)\left(x-x_{k+1}\right)$$

取上式中的$$x = x_k$$，得：

$$f^{\prime}\left(x_{k}\right)=f^{\prime}\left(x_{k+1}\right)+f^{\prime \prime}\left(x_{k+1}\right)\left(x-x_{k+1}\right)$$

即：

$$g_{k+1}-g_{k}=H_{k+1} \cdot\left(x_{k+1}-x_{k}\right)$$

可得：

$$H_{k+1}^{-1} \cdot\left(g_{k+1}-g_{k}\right)=x_{k+1}-x_{k}$$

上面这个式子称为**“拟牛顿条件”**，由它来对Hessen矩阵做约束。

