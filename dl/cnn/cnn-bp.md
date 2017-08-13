# 卷积神经网络\(CNN\)反向传播算法

---

在[卷积神经网络\(CNN\)前向传播算法](/dl/cnn/cnn-fp.md)中，我们对CNN的前向传播算法做了总结，基于CNN前向传播算法的基础，我们下面就对CNN的反向传播算法做一个总结。在阅读本文前，建议先研究DNN的反向传播算法：[深度神经网络（DNN）反向传播算法\(BP\)](/dl/introduction/dnn-bp.md)

# 1. 回顾DNN的反向传播算法

我们首先回顾DNN的反向传播算法。在DNN中，我们是首先计算出输出层的$$\delta^L:\delta^L = \frac{\partial J(W,b)}{\partial z^L} = \frac{\partial J(W,b)}{\partial a^L}\odot \sigma^{'}(z^L)$$

利用数学归纳法，用$$\delta^{l+1}$$的值一步步的向前求出第l层的$$\delta^l$$，表达式为：$$\delta^{l} = \delta^{l+1}\frac{\partial z^{l+1}}{\partial z^{l}} = (W^{l+1})^T\delta^{l+1}\odot \sigma^{'}(z^l)$$

有了$$\delta^l$$的表达式，从而求出W,b的梯度表达式：

$$\frac{\partial J(W,b)}{\partial W^l} = \frac{\partial J(W,b,x,y)}{\partial z^l} \frac{\partial z^l}{\partial W^l} = \delta^{l}(a^{l-1})^T$$

$$\frac{\partial J(W,b,x,y)}{\partial b^l} = \frac{\partial J(W,b)}{\partial z^l} \frac{\partial z^l}{\partial b^l} = \delta^{l}$$

有了W,b梯度表达式，就可以用梯度下降法来优化W,b,求出最终的所有W,b的值。

现在我们想把同样的思想用到CNN中，很明显，CNN有些不同的地方，不能直接去套用DNN的反向传播算法的公式。

# 2. CNN的反向传播算法思想

要套用DNN的反向传播算法到CNN，有几个问题需要解决：

1）池化层没有激活函数，这个问题倒比较好解决，我们可以令池化层的激活函数为$$\sigma(z) $$= z，即激活后就是自己本身。这样池化层激活函数的导数为1.

2）池化层在前向传播的时候，对输入进行了压缩，那么我们现在需要向前反向推导$$\delta^{l-1}$$，这个推导方法和DNN完全不同。

3\) 卷积层是通过张量卷积，或者说若干个矩阵卷积求和而得的当前层的输出，这和DNN很不相同，DNN的全连接层是直接进行矩阵乘法得到当前层的输出。这样在卷积层反向传播的时候，上一层的$$\delta^{l-1}$$递推计算方法肯定有所不同。

4）对于卷积层，由于W使用的运算是卷积，那么从$$\delta^l$$推导出该层的所有卷积核的W,b的方式也不同。

从上面可以看出，问题1比较好解决，但是问题2,3,4就需要好好的动一番脑筋了，而问题2,3,4也是解决CNN反向传播算法的关键所在。另外大家要注意到的是，DNN中的$$a_l,z_l$$都只是一个向量，而我们CNN中的$$a_l,z_l$$都是一个张量，这个张量是三维的，即由若干个输入的子矩阵组成。

下面我们就针对问题2,3,4来一步步研究CNN的反向传播算法。

在研究过程中，需要注意的是，由于卷积层可以有多个卷积核，各个卷积核的处理方法是完全相同且独立的，为了简化算法公式的复杂度，我们下面提到卷积核都是卷积层中若干卷积核中的一个。

# 3. 已知池化层的$$\delta^l$$，推导上一隐藏层的$$\delta^{l-1}$$

我们首先解决上面的问题2，如果已知池化层的$$\delta^l$$，推导出上一隐藏层的$$\delta^{l-1}$$。

在前向传播算法时，池化层一般我们会用MAX或者Average对输入进行池化，池化的区域大小已知。现在我们反过来，要从缩小后的误差$$\delta^l$$，还原前一次较大区域对应的误差。

在反向传播时，我们首先会把$$\delta^l$$的所有子矩阵大小还原成池化之前的大小，然后如果是MAX，则把$$\delta^l$$的所有子矩阵的各个池化局域的值放在之前做前向传播算法得到最大值的位置。如果是Average，则把$$\delta^l$$的所有子矩阵的各个池化局域的值取平均后放在还原后的子矩阵位置。这个过程一般叫做upsample。

用一个例子可以很方便的表示：假设我们的池化区域大小是2x2。$$\delta^l$$的第k个子矩阵为:$$\delta_k^l = \left( \begin{array}{ccc} 2& 8 \\ 4& 6 \end{array} \right)$$

由于池化区域为2x2，我们先讲$$\delta_k^l$$做还原，即变成：$$\left( \begin{array}{ccc} 0&0&0&0 \\ 0&2& 8&0 \\ 0&4&6&0 \\ 0&0&0&0 \end{array} \right)$$

如果是MAX，假设我们之前在前向传播时记录的最大值位置分别是左上，右下，右上，左下，则转换后的矩阵为：$$\left( \begin{array}{ccc} 2&0&0&0 \\ 0&0& 0&8 \\ 0&4&0&0 \\ 0&0&6&0 \end{array} \right)$$

如果是Average，则进行平均：转换后的矩阵为：$$\left( \begin{array}{ccc} 0.5&0.5&2&2 \\ 0.5&0.5&2&2 \\ 1&1&1.5&1.5 \\ 1&1&1.5&1.5 \end{array} \right)$$

这样我们就得到了上一层 $$\frac{\partial J(W,b)}{\partial a_k^{l-1}}$$的值，要得到$$\delta_k^{l-1}$$：$$\delta_k^{l-1} = \frac{\partial J(W,b)}{\partial a_k^{l-1}} \frac{\partial  a_k^{l-1}}{\partial z_k^{l-1}} = upsample(\delta_k^l) \odot \sigma^{'}(z_k^{l-1})$$

其中，upsample函数完成了池化误差矩阵放大与误差重新分配的逻辑。

我们概括下，对于张量$$\delta^{l-1}$$，我们有：$$\delta^{l-1} =  upsample(\delta^l) \odot \sigma^{'}(z^{l-1})$$

# 4. 已知卷积层的$$\delta^l$$，推导上一隐藏层的$$\delta^{l-1}$$

对于卷积层的反向传播，我们首先回忆下卷积层的前向传播公式：$$a^l= \sigma(z^l) = \sigma(a^{l-1}*W^l +b^l)$$

其中$$n\_in$$为上一隐藏层的输入子矩阵个数。

在DNN中，我们知道$$\delta^{l-1}$$和$$\delta^{l}$$的递推关系为：$$\delta^{l} = \frac{\partial J(W,b)}{\partial z^l} = \frac{\partial J(W,b)}{\partial z^{l+1}}\frac{\partial z^{l+1}}{\partial z^{l}} = \delta^{l+1}\frac{\partial z^{l+1}}{\partial z^{l}}$$

因此要推导出$$\delta^{l-1}$$和$$\delta^{l}$$的递推关系，必须计算$$\frac{\partial z^{l}}{\partial z^{l-1}}$$的梯度表达式。

注意到$$z^{l}$$和$$z^{l-1}$$的关系为：$$z^l = a^{l-1}*W^l +b^l =\sigma(z^{l-1})*W^l +b^l$$

因此我们有：$$\delta^{l-1} =  \delta^{l}\frac{\partial z^{l}}{\partial z^{l-1}} = \delta^{l}*rot180(W^{l}) \odot  \sigma^{'}(z^{l-1})$$

这里的式子其实和DNN的类似，区别在于对于含有卷积的式子求导时，卷积核被旋转了180度。即式子中的$$rot180()$$，翻转180度的意思是上下翻转一次，接着左右翻转一次。在DNN中这里只是矩阵的转置。那么为什么呢？由于这里都是张量，直接推演参数太多了。我们以一个简单的例子说明为啥这里求导后卷积核要翻转。

假设我们l-1层的输出$$a^{l-1}$$是一个3x3矩阵，第l层的卷积核$$W^l$$是一个2x2矩阵，采用1像素的步幅，则输出$$z^{l}$$是一个2x2的矩阵。我们简化$$b^l$$都是0,则有$$a^{l-1}*W^l = z^{l}$$

我们列出a,W,z的矩阵表达式如下：$$\left( \begin{array}{ccc} a_{11}&a_{12}&a_{13} \\ a_{21}&a_{22}&a_{23}\\ a_{31}&a_{32}&a_{33} \end{array} \right)    *  \left( \begin{array}{ccc} w_{11}&w_{12}\\ w_{21}&w_{22} \end{array} \right) = \left( \begin{array}{ccc} z_{11}&z_{12}\\ z_{21}&z_{22} \end{array} \right)$$

利用卷积的定义，很容易得出：

$$z_{11} = a_{11}w_{11} + a_{12}w_{12} + a_{21}w_{21} +   a_{22}w_{22}$$

$$z_{12} = a_{12}w_{11} + a_{13}w_{12} + a_{22}w_{21} +   a_{23}w_{22}$$

$$z_{21} = a_{21}w_{11} + a_{22}w_{12} + a_{31}w_{21} +   a_{32}w_{22}$$

$$z_{22} = a_{22}w_{11} + a_{23}w_{12} + a_{32}w_{21} +   a_{33}w_{22}$$

接着我们模拟反向求导：$$\nabla a^{l-1} = \frac{\partial J(W,b)}{\partial a^{l-1}} = \frac{\partial J(W,b)}{\partial z^{l}} \frac{\partial z^{l}}{\partial a^{l-1}} = \delta^{l} \frac{\partial z^{l}}{\partial a^{l-1}}$$

从上式可以看出，对于$$a^{l-1}$$的梯度误差$$\nabla a^{l-1}$$，等于第l层的梯度误差乘以$$\frac{\partial z^{l}}{\partial a^{l-1}}$$，而$$\frac{\partial z^{l}}{\partial a^{l-1}}$$对应上面的例子中相关联的w的值。假设我们的z矩阵对应的反向传播误差是$$\delta_{11}, \delta_{12}, \delta_{21}, \delta_{22}$$组成的2x2矩阵，则利用上面梯度的式子和4个等式，我们可以分别写出$$\nabla a^{l-1}$$的9个标量的梯度。

比如对于$$a_{11}$$的梯度，由于在4个等式中$$a_{11}$$只和$$z_{11}$$有乘积关系，从而我们有：$$\nabla a_{11} = \delta_{11}w_{11}$$

对于$$a_{12}$$的梯度，由于在4个等式中$$a_{12}$$和$$z_{12}, z_{11}$$有乘积关系，从而我们有：$$\nabla a_{12} = \delta_{11}w_{12} + \delta_{12}w_{11}$$

同样的道理我们得到：

$$\nabla a_{13} = \delta_{12}w_{12}$$

$$\nabla a_{21} = \delta_{11}w_{21} + \delta_{21}w_{11}$$

$$\nabla a_{22} = \delta_{11}w_{22} + \delta_{12}w_{21} + \delta_{21}w_{12} + \delta_{22}w_{11}$$

$$\nabla a_{23} = \delta_{12}w_{22} + \delta_{22}w_{12}$$

$$\nabla a_{31} = \delta_{21}w_{21}$$

$$\nabla a_{32} = \delta_{21}w_{22} + \delta_{22}w_{21}$$

$$\nabla a_{33} = \delta_{22}w_{22}$$

这上面9个式子其实可以用一个矩阵卷积的形式表示，即：

$$\left( \begin{array}{ccc} 0&0&0&0 \\ 0&\delta_{11}& \delta_{12}&0 \\ 0&\delta_{21}&\delta_{22}&0 \\ 0&0&0&0 \end{array} \right) * \left( \begin{array}{ccc} w_{22}&w_{21}\\ w_{12}&w_{11} \end{array} \right)  = \left( \begin{array}{ccc} \nabla a_{11}&\nabla a_{12}&\nabla a_{13} \\ \nabla a_{21}&\nabla a_{22}&\nabla a_{23}\\ \nabla a_{31}&\nabla a_{32}&\nabla a_{33} \end{array} \right)$$

为了符合梯度计算，我们在误差矩阵周围填充了一圈0，此时我们将卷积核翻转后和反向传播的梯度误差进行卷积，就得到了前一次的梯度误差。这个例子直观的介绍了为什么对含有卷积的式子求导时，卷积核要翻转180度的原因。

以上就是卷积层的误差反向传播过程。

# 5. 已知卷积层的$$\delta^l$$，推导该层的W,b的梯度

好了，我们现在已经可以递推出每一层的梯度误差$$\delta^l$$了，对于全连接层，可以按DNN的反向传播算法求该层W,b的梯度，而池化层并没有W,b,也不用求W,b的梯度。只有卷积层的W,b需要求出。

注意到卷积层z和W,b的关系为：$$z^l = a^{l-1}*W^l +b$$

因此我们有：$$\frac{\partial J(W,b)}{\partial W^{l}} = \frac{\partial J(W,b)}{\partial z^{l}}\frac{\partial z^{l}}{\partial W^{l}} = \delta^l * rot180( a^{l-1})$$

由于我们有上一节的基础，大家应该清楚为什么这里求导后$$a^{l-1}$$要旋转180度了。

而对于b,则稍微有些特殊，因为$$\delta^l$$是三维张量，而b只是一个向量，不能像DNN那样直接和$$\delta^l$$相等。通常的做法是将$$\delta^l$$的各个子矩阵的项分别求和，得到一个误差向量，即为b的梯度：$$\frac{\partial J(W,b)}{\partial b^{l}} = \sum\limits_{u,v}(\delta^l)_{u,v}$$

# 6. CNN反向传播算法总结

现在我们总结下CNN的反向传播算法，以最基本的批量梯度下降法为例来描述反向传播算法。

输入：m个图片样本，CNN模型的层数L和所有隐藏层的类型，对于卷积层，要定义卷积核的大小K，卷积核子矩阵的维度F，填充大小P，步幅S。对于池化层，要定义池化区域大小k和池化标准（MAX或Average），对于全连接层，要定义全连接层的激活函数（输出层除外）和各层的神经元个数。梯度迭代参数迭代步长$$\alpha$$,最大迭代次数MAX与停止迭代阈值$$\epsilon$$

输出：CNN模型各隐藏层与输出层的W,b

1\) 初始化各隐藏层与输出层的各W,b的值为一个随机值。

2）for iter from 1 to MAX：

2-1\) for i =1 to m：

a\) 将CNN输入$$a^1$$设置为$$x_i$$对应的张量

b\) for l=2 to L-1，根据下面3种情况进行前向传播算法计算：

b-1\) 如果当前是全连接层：则有$$a^{i,l} = \sigma(z^{i,l}) = \sigma(W^la^{i,l-1} + b^{i,l})$$

b-2\) 如果当前是卷积层：则有$$a^{i,l} = \sigma(z^{i,l}) = \sigma(W^l*a^{i,l-1} + b^{i,l})$$

b-3\) 如果当前是池化层：则有$$a^{i,l}= pool(a^{i,l-1})$$, 这里的pool指按照池化区域大小k和池化标准将输入张量缩小的过程。

c\) 对于输出层第L层:$$a^{i,L}= softmax(z^{i,L}) = softmax(W^{i,L}a^{i,L-1} +b^{i,L})$$

c\) 通过损失函数计算输出层的$$\delta^{i,L}$$

d\) for l= L to 2, 根据下面3种情况进行进行反向传播算法计算:

d-1\)  如果当前是全连接层：$$\delta^{i,l} =  (W^{l+1})^T\delta^{i,l+1}\odot \sigma^{'}(z^{i,l})$$

d-2\) 如果当前是卷积层：$$\delta^{i,l} = \delta^{i,l+1}*rot180(W^{l+1}) \odot  \sigma^{'}(z^{i,l})$$

d-3\) 如果当前是池化层：$$\delta^{i,l} =  upsample(\delta^{i,l+1}) \odot \sigma^{'}(z^{i,l})$$

2-2\) for l = 2 to L，根据下面2种情况更新第l层的$$W^l,b^l$$:

2-2-1\) 如果当前是全连接层：$$W^l = W^l -\alpha \sum\limits_{i=1}^m \delta^{i,l}(a^{i, l-1})^T$$ $$b^l = b^l -\alpha \sum\limits_{i=1}^m \delta^{i,l}$$

2-2-2\) 如果当前是卷积层，对于每一个卷积核有：$$W^l = W^l -\alpha \sum\limits_{i=1}^m \delta^{i,l}* rot180(a^{i, l-1}), b^l = b^l -\alpha \sum\limits_{i=1}^m \sum\limits_{u,v}(\delta^{i,l})_{u,v}$$

2-3\) 如果所有W，b的变化值都小于停止迭代阈值$$\epsilon$$，则跳出迭代循环到步骤3。

3） 输出各隐藏层与输出层的线性关系系数矩阵W和偏倚向量b。

