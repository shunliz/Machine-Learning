# 深度神经网络（DNN）反向传播算法\(BP\)

---

# 1. DNN反向传播算法要解决的问题

在了解DNN的反向传播算法前，我们先要知道DNN反向传播算法要解决的问题，也就是说，什么时候我们需要这个反向传播算法？

回到我们监督学习的一般问题，假设我们有m个训练样本：$${(x_1,y_1), (x_2,y_2), ..., (x_m,y_m)}$$,其中x为输入向量，特征维度为$$n_{in}$$,而y为输出向量，特征维度为$$n_{out}$$_。_我们需要利用这m个样本训练出一个模型，当有一个新的测试样本$$(x_{test},?)$$来到时, 我们可以预测$$y_{test}$$向量的输出。

如果我们采用DNN的模型，即我们使输入层有$$n\_in$$个神经元，而输出层有$$n\_out$$个神经元。再加上一些含有若干神经元的隐藏层。此时我们需要找到合适的所有隐藏层和输出层对应的线性系数矩阵W,偏倚向量b,让所有的训练样本输入计算出的输出尽可能的等于或很接近样本输出。怎么找到合适的参数呢？

如果大家对传统的机器学习的算法优化过程熟悉的话，这里就很容易联想到我们可以用一个合适的损失函数来度量训练样本的输出损失，接着对这个损失函数进行优化求最小化的极值，对应的一系列线性系数矩阵W,偏倚向量b即为我们的最终结果。在DNN中，损失函数优化极值求解的过程最常见的一般是通过梯度下降法来一步步迭代完成的，当然也可以是其他的迭代方法比如牛顿法与拟牛顿法。如果大家对梯度下降法不熟悉，建议先阅读我之前写的[梯度下降（Gradient Descent）小结](http://www.cnblogs.com/pinard/p/5970503.html)。

对DNN的损失函数用梯度下降法进行迭代优化求极小值的过程即为我们的反向传播算法。

# 2. DNN反向传播算法的基本思路

在进行DNN反向传播算法前，我们需要选择一个损失函数，来度量训练样本计算出的输出和真实的训练样本输出之间的损失。你也许会问：训练样本计算出的输出是怎么得来的？这 个输出是随机选择一系列W,b,用我们上一节的前向传播算法计算出来的。即通过一系列的计算：$$a^l = \sigma(z^l) = \sigma(W^la^{l-1} + b^l)$$。计算到输出层第L层对应的$$a^L$$即为前向传播算法计算出来的输出。

回到损失函数，DNN可选择的损失函数有不少，为了专注算法，这里我们使用最常见的均方差来度量损失。即对于每个样本，我们期望最小化下式：$$J(W,b,x,y) = \frac{1}{2}||a^L-y||_2^2$$

其中，$$a^L$$和y为特征维度为$$n\_out$$的向量,而$$||S||_2$$为S的L2范数。

损失函数有了，现在我们开始用梯度下降法迭代求解每一层的W,b。

首先是输出层第L层。注意到输出层的W,b满足下式：$$a^L = \sigma(z^L) = \sigma(W^La^{L-1} + b^L)$$

这样对于输出层的参数，我们的损失函数变为：$$J(W,b,x,y) = \frac{1}{2}||a^L-y||_2^2 =  \frac{1}{2}|| \sigma(W^La^{L-1} + b^L)-y||_2^2$$

这样求解W,b的梯度就简单了：$$\frac{\partial J(W,b,x,y)}{\partial W^L} = \frac{\partial J(W,b,x,y)}{\partial z^L}\frac{\partial z^L}{\partial W^L} =(a^L-y) (a^{L-1})^T\odot \sigma^{'}(z)$$$$\frac{\partial J(W,b,x,y)}{\partial b^L} = \frac{\partial J(W,b,x,y)}{\partial z^L}\frac{\partial z^L}{\partial b^L} =(a^L-y)\odot \sigma^{'}(z^L)$$

注意上式中有一个符号$$\odot$$,它代表Hadamard积，对于两个维度相同的向量A$$（a_1,a_2,...a_n）^T$$和B$$（b_1,b_2,...b_n）^T$$,则$$A \odot B = (a_1b_1, a_2b_2,...a_nb_n)^T$$。

我们注意到在求解输出层的W,b的时候，有公共的部分$$\frac{\partial J(W,b,x,y)}{\partial z^L}$$，因此我们可以把公共的部分即对$$z^L$$先算出来，记为：$$\delta^L = \frac{\partial J(W,b,x,y)}{\partial z^L} = (a^L-y)\odot \sigma^{'}(z^L)$$

现在我们终于把输出层的梯度算出来了，那么如何计算上一层L-1层的梯度，上上层L-2层的梯度呢？这里我们需要一步步的递推，注意到对于第l层的未激活输出$$z^l$$，它的梯度可以表示为:$$\delta^l =\frac{\partial J(W,b,x,y)}{\partial z^l} = \frac{\partial J(W,b,x,y)}{\partial z^L}\frac{\partial z^L}{\partial z^{L-1}}\frac{\partial z^{L-1}}{\partial z^{L-2}}...\frac{\partial z^{l+1}}{\partial z^{l}}$$

如果我们可以依次计算出第l层的$$\delta^l$$,则该层的$$W^l,b^l$$很容易计算？为什么呢？注意到根据前向传播算法，我们有：$$z^l= W^la^{l-1} + b^l$$

所以根据上式我们可以很方便的计算出第l层的$$W^l,b^l$$的梯度如下：$$\frac{\partial J(W,b,x,y)}{\partial W^l} = \frac{\partial J(W,b,x,y)}{\partial z^l} \frac{\partial z^l}{\partial W^l} = \delta^{l}(a^{l-1})^T\frac{\partial J(W,b,x,y)}{\partial b^l} = \frac{\partial J(W,b,x,y)}{\partial z^l} \frac{\partial z^l}{\partial b^l} = \delta^{l}$$

那么现在问题的关键就是要求出$$\delta^{l}$$了。这里我们用数学归纳法，第L层的$$\delta^{L}$$上面我们已经求出， 假设第l+1层的$$\delta^{l+1}$$已经求出来了，那么我们如何求出第l层的\delta^{l}呢？我们注意到：$$\delta^{l} = \frac{\partial J(W,b,x,y)}{\partial z^l} = \frac{\partial J(W,b,x,y)}{\partial z^{l+1}}\frac{\partial z^{l+1}}{\partial z^{l}} = \delta^{l+1}\frac{\partial z^{l+1}}{\partial z^{l}}$$

可见，用归纳法递推$$\delta^{l+1}$$和$$\delta^{l}$$的关键在于求解$$\frac{\partial z^{l+1}}{\partial z^{l}}$$。

而$$z^{l+1}$$和$$z^{l}$$的关系其实很容易找出：$$z^{l+1}= W^{l+1}a^{l} + b^{l+1} = W^{l+1}\sigma(z^l) + b^{l+1}$$

这样很容易求出：$$\frac{\partial z^{l+1}}{\partial z^{l}} = (W^{l+1})^T\odot \sigma^{'}(z^l)$$

将上式带入上面$$\delta^{l+1}$$和$$\delta^{l}$$关系式我们得到：$$\delta^{l} = \delta^{l+1}\frac{\partial z^{l+1}}{\partial z^{l}} = (W^{l+1})^T\delta^{l+1}\odot \sigma^{'}(z^l)$$

现在我们得到了$$\delta^{l}$$的递推关系式，只要求出了某一层的$$\delta^{l}$$，求解$$W^l,b^l$$的对应梯度就很简单的。

# 3. DNN反向传播算法过程

现在我们总结下DNN反向传播算法的过程。由于梯度下降法有批量（Batch），小批量\(mini-Batch\)，随机三个变种，为了简化描述，这里我们以最基本的批量梯度下降法为例来描述反向传播算法。实际上在业界使用最多的是mini-Batch的梯度下降法。不过区别仅仅在于迭代时训练样本的选择而已。

输入: 总层数L，以及各隐藏层与输出层的神经元个数，激活函数，损失函数，迭代步长$$\alpha$$,最大迭代次数MAX与停止迭代阈值$$\epsilon$$，输入的m个训练样本$$\{(x_1,y_1), (x_2,y_2), ..., (x_m,y_m)\}$$

输出：各隐藏层与输出层的线性关系系数矩阵W和偏倚向量b

1\) 初始化各隐藏层与输出层的线性关系系数矩阵W和偏倚向量b的值为一个随机值。

2）for iter to 1 to MAX：

2-1\) for i =1 to m：

a\) 将DNN输入$$a^1$$设置为$$x_i$$

b\) for l=2 to L，进行前向传播算法计算$$a^{i,l} = \sigma(z^{i,l}) = \sigma(W^la^{i,l-1} + b^l)$$

c\) 通过损失函数计算输出层的$$\delta^{i,L}$$

d\) for l= L to 2, 进行反向传播算法计算$$\delta^{i,l} =  (W^{l+1})^T\delta^{i,l+1}\odot \sigma^{'}(z^{i,l})$$

2-2\) for l = 2 to L，更新第l层的$$W^l,b^l$$:

$$W^l = W^l -\alpha \sum\limits_{i=1}^m \delta^{i,l}(a^{i, l-1})^T$$

$$b^l = b^l -\alpha \sum\limits_{i=1}^m \delta^{i,l}$$

2-3\) 如果所有W，b的变化值都小于停止迭代阈值$$\epsilon$$，则跳出迭代循环到步骤3。

3） 输出各隐藏层与输出层的线性关系系数矩阵W和偏倚向量b。

