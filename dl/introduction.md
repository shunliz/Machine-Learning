# 绪论 {#绪论}

## 什么是深度学习 {#什么是深度学习}

---

简单的说明这本书是一本入门的书籍，跳过不想关的内容，直接进入深度学习的发展趋势。附上使用的连接 www. deeplearningbook.org.[这里可以查看目录连接](http://www.deeplearningbook.org/contents/intro.html)这本书的主要目的是去解决这种更加需要直觉的问题。解决这样的一个问题就可以使得计算机从经验中学习并且从层级的概念上去理解这个世界。层级的概念使得计算机可以用简单的概念去学习复杂的概念。如果采用一个graph来描述这些概念是怎么在彼此的上面构成的，这样的graph就是deep的，所以就把这种方式称为了 AI Deep Learning。计算机想要智能的方式就是需要找到一个方式将这种normal的知识转换到计算机当中去。 有几个人工智能项目想采用硬编码的方式来实现（硬编码是指将可变变量用一个固定值来代替的方法）但是都没有成功。这种方式通常被称为通向人工智能的knowledge base方式。著名的项目之一就是Cyc。但是这样的失败是有意的，通过这些尝试，人们认为AI需要通过从raw data提取模式从而获得它自己的知识，所以有了Machine Learning。最简单的机器学习方法之一就是逻辑回归（logistics regression）用来判断是否建议剖腹产，而naive Bayes（朴素贝叶斯方法）可以用来区分正常邮件和垃圾邮件。 简单的机器学习算法很大程度上依赖与所给数据的表述能力（representation），例如上述的逻辑回归例子，并不能直接用于病人的检测，而是需要依靠doctor所给出的数据。这个表述当中所包含的信息块就被称作为feature（特征）。逻辑回归就是学习这些feature和各种输出之间的相关性，但是它不能影响features的定义（Mu：也就是说这个方法不能自主的选择合适的特征）。客观的说，很多的问题都可以通过设计合适的特征来完成。 但是（说道但是，对吧，就到核心了，也即是为什么会有DL，也就说明了DL的特点是什么）对于很多任务来讲，我们并不知道设计怎样的特征算是好的合适的特征。解决这个问题的方法就是找到一中机器学习算法，这个方法不仅能发现映射表述到输出，同时能够发现表述本身，这个方式就被称为representation learning\(学习\)。学习到的特征一般比手动设计的特征具有更好的效果。那么关于这种表述学习的一个比较好的例子就是autoencoder（计算机自动编码，encoder+decoder）。 在设计学习特征的特征以及算法的时候，我们的目标是将描述观测数据的变量因素（factor of variation）分开，采用factores来描述不同因素的影响。真实世界的人工智能的主要困难应用之处在于很多的变量因素会影响到我们观测数据的每一个片信息（piece of data）。DL解决这个问题的方式就是就是采用其他更加简单的representations来表示，当然这个是由Dl学习所得到的，下图是一个典型的图像识别深度网络结构,DL在这个方面就是将一个复杂的mapping划分称为一系列嵌套的简单映射\(nested simple mappings）,每个都可以模型的不同层来进行表示。深度网络的典型结构例子就是前馈深度网络（deep network）或者称为多层感知器\(multilayer perceptron,MLP）（概念：Depth is the length of the longest path from input tooutput but depends on the deﬁnition of what constitutes a possible computational step）![](https://mujanfun.gitbooks.io/deeplearning-google/content/%E6%8D%95%E8%8E%B7.PNG "表示了一个深度学习结构是如何描述一幅人的图像")Figure 1

```
有两种方式来衡量模型的深度：第一种就是sequential instructions的数目，我们可以把这个想象成最长的计算路径；另一种方式就是描述概念之间相互关系的网络深度，但是这个方式呢要计算需要计算每个concept的representation，所以会比graph的深度要深，主要是因为简单的概念能被定义，从而能够表述更加复杂的概念。（Nor is there a consensus about how much depth a model requires to qualify as “deep.”）.Figure2 描述了机器学习的关系，Figure5描述了具体是怎么工作的。![机器学习之间的关系](F2.PNG)
![各个机器学习方法怎么使用的](F3.PNG)

```

## 深度学习的历史 {#深度学习的历史}

```
* 深度学习有着长而丰富的开始，曾经有过很多的名字，作为深度学习受到了普遍的接收和欢迎
* 因为训练数据的增加导致了深度学习越来越有用
* 随着计算机技术的增长，深度学习模型规模也越来越大
* 随着时间的推移，深度学习可以解决越来越复杂的问题

```

深度学习的历史大致可以分为三个阶段：1940-1960，这二十年间，主要是cybernetics（控制论）的发展；1980-1990，这十年间主要是connectionism发展；再其次就是自2006年以来deep Learning（深度学习）的发展，但是直到2016才在书中出现。 最早的一些学习算法是基于生物学习得到的，所以有了早期的ANN，下图是这几十年这个领域的发展趋势图![](https://mujanfun.gitbooks.io/deeplearning-google/content/f4.PNG "发展趋势")深度学习并不是完全的模拟人脑，二十融合了线性代数、统计学、信息论、数值优化的等等。其实在这个领域，有人关注神经学，也有人不关注神经学。关注神经学的就成为计算神经学\(computational neuroscience\),是和深度学习不相同的研究领域，计算神经学主要关注怎么精确的模拟人脑的工作过程。 深度学习的第二阶段是1980s,主要是伴随着认知科学（cognitive science）的兴起。联接机制就是将大量的简单的计算单元用网络联接在一起，从而达到智能化的表现。这一阶段为当前的深度学习留下了一些核心的概念：分布式特征表达；反向传播算法；long short-term memory\(LSTM\)方法。 Kernel machines和graphical model 导致了神经网络的衰退直到2007年。 深度学习需要的是用一个单独的深度学习结构来解决多种不同的问题。2006年，Geoffrey Hinton展示了DBF，采用greedy layer-wise进行有效的训练。Dl的使用，不仅是用来强调可以训练比以前更深的网络，更是将注意力集中在了深度的理论重要性。 其他的还有数据的增加和有更多的计算资源导致的模型size的增加，提高准确定、复杂性和现实世界的影响。重要的是强化学习，通过尝试和失误，在不要人参与的前提下可以达到学习的目的，深度学习可以用来改善强化学习的能力。

##  {#一-单层神经网络感知器}

## **一 单层神经网络（感知器）** {#一-单层神经网络感知器}

**1.结构**

下面来说明感知器模型。

在原来MP模型的“输入”位置添加神经元节点，标志其为“输入单元”。其余不变，于是我们就有了下图：从本图开始，我们将权值w1, w2, w3写到“连接线”的中间。

![](http://img.blog.csdn.net/20160114193502846)  
                                                      图1 单层神经网络

在“感知器”中，有两个层次。分别是输入层和输出层。输入层里的“输入单元”只负责传输数据，不做计算。输出层里的“输出单元”则需要对前面一层的输入进行计算。

我们把需要计算的层次称之为“计算层”，并把拥有一个计算层的网络称之为“单层神经网络”。有一些文献会按照网络拥有的层数来命名，例如把“感知器”称为两层神经网络。但在本文里，我们根据计算层的数量来命名。

假如我们要预测的目标不再是一个值，而是一个向量，例如\[2,3\]。那么可以在输出层再增加一个“输出单元”。

下图显示了带有两个输出单元的单层神经网络，其中输出单元z1的计算公式如下图。

![](http://img.blog.csdn.net/20160114193621726 "神经网络")

```
                                         图2 单层神经网络\(Z1\)
```

可以看到，z1的计算跟原先的z并没有区别。

我们已知一个神经元的输出可以向多个神经元传递，因此z2的计算公式如下图。

![](http://img.blog.csdn.net/20160114193659368 "神经网络")

```
                                         图3 单层神经网络\(Z2\)
```

可以看到，z2的计算中除了三个新的权值：w4，w5，w6以外，其他与z1是一样的。

整个网络的输出如下图。

![](http://img.blog.csdn.net/20160114193852865 "神经网络")

```
                                      图4 单层神经网络\(Z1和Z2\)
```

目前的表达公式有一点不让人满意的就是：w4，w5，w6是后来加的，很难表现出跟原先的w1，w2，w3的关系。

因此我们改用二维的下标，用wx,y来表达一个权值。下标中的x代表后一层神经元的序号，而y代表前一层神经元的序号（序号的顺序从上到下）。

例如，w1,2代表后一层的第1个神经元与前一层的第2个神经元的连接的权值（这种标记方式参照了Andrew Ng的课件）。根据以上方法标记，我们有了下图。

![](http://img.blog.csdn.net/20160114193834036 "神经网络")

```
                                      图5 单层神经网络\(扩展\)
```

如果我们仔细看输出的计算公式，会发现这两个公式就是线性代数方程组。因此可以用矩阵乘法来表达这两个公式。

例如，输入的变量是\[a1，a2，a3\]T（代表由a1，a2，a3组成的列向量），用向量a来表示。方程的左边是\[z1，z2\]T，用向量z来表示。

系数则是矩阵W（2行3列的矩阵，排列形式与公式中的一样）。

于是，输出公式可以改写成：

g\(W \* a\) = z;

这个公式就是神经网络中从前一层计算后一层的矩阵运算。

**2.效果**

与神经元模型不同，感知器中的权值是通过训练得到的。因此，根据以前的知识我们知道，感知器类似一个逻辑回归模型，可以做线性分类任务。

我们可以用决策分界来形象的表达分类的效果。决策分界就是在二维的数据平面中划出一条直线，当数据的维度是3维的时候，就是划出一个平面，当数据的维度是n维时，就是划出一个n-1维的超平面。

下图显示了在二维平面中划出决策分界的效果，也就是感知器的分类效果。

![](http://img.blog.csdn.net/20160114193946318 "神经网络")

```
                                图6 单层神经网络（决策分界）
```

## **二 两层神经网络（多层感知器）** {#二-两层神经网络多层感知器}

**1.结构**

两层神经网络除了包含一个输入层，一个输出层以外，还增加了一个中间层。此时，中间层和输出层都是计算层。我们扩展上节的单层神经网络，在右边新加一个层次（只含有一个节点）。

现在，我们的权值矩阵增加到了两个，我们用上标来区分不同层次之间的变量。

例如ax\(y\)代表第y层的第x个节点。z1，z2变成了a1\(2\)，a2\(2\)。下图给出了a1\(2\)，a2\(2\)的计算公式。

![](http://img.blog.csdn.net/20160114194120339 "神经网络")

```
                                图7 两层神经网络（中间层计算）
```

计算最终输出z的方式是利用了中间层的a1\(2\)，a2\(2\)和第二个权值矩阵计算得到的，如下图。

![](http://img.blog.csdn.net/20160114194215782 "神经网络")

```
                             图8 两层神经网络（输出层计算）
```

假设我们的预测目标是一个向量，那么与前面类似，只需要在“输出层”再增加节点即可。

我们使用向量和矩阵来表示层次中的变量。a\(1\)，a\(2\)，z是网络中传输的向量数据。W\(1\)和W\(2\)是网络的矩阵参数。如下图。

![](http://img.blog.csdn.net/20160114194251724 "神经网络")

```
                                   图9 两层神经网络（向量形式）
```

使用矩阵运算来表达整个计算公式的话如下：

g\(W\(1\) \* a\(1\)\) = a\(2\);

g\(W\(2\) \* a\(2\)\) = z;

由此可见，使用矩阵运算来表达是很简洁的，而且也不会受到节点数增多的影响（无论有多少节点参与运算，乘法两端都只有一个变量）。因此神经网络的教程中大量使用矩阵运算来描述。

需要说明的是，至今为止，我们对神经网络的结构图的讨论中都没有提到偏置节点（bias unit）。事实上，这些节点是默认存在的。它本质上是一个只含有存储功能，且存储值永远为1的单元。在神经网络的每个层次中，除了输出层以外，都会含有这样一个偏置单元。正如线性回归模型与逻辑回归模型中的一样。

偏置单元与后一层的所有节点都有连接，我们设这些参数值为向量b，称之为偏置。如下图。

![](http://img.blog.csdn.net/20160114194326476 "神经网络")

```
                                   图10 两层神经网络（考虑偏置节点）
```

可以看出，偏置节点很好认，因为其没有输入（前一层中没有箭头指向它）。有些神经网络的结构图中会把偏置节点明显画出来，有些不会。一般情况下，我们都不会明确画出偏置节点。

在考虑了偏置以后的一个神经网络的矩阵运算如下：

g\(W\(1\) \* a\(1\) + b\(1\)\) = a\(2\);

g\(W\(2\) \* a\(2\) + b\(2\)\) = z;

需要说明的是，在两层神经网络中，我们不再使用sgn函数作为函数g，而是使用平滑函数sigmoid作为函数g。我们把函数g也称作激活函数（active function）。

事实上，神经网络的本质就是通过参数与激活函数来拟合特征与目标之间的真实函数关系。初学者可能认为画神经网络的结构图是为了在程序中实现这些圆圈与线，但在一个神经网络的程序中，既没有“线”这个对象，也没有“单元”这个对象。实现一个神经网络最需要的是线性代数库。

**2.效果**

与单层神经网络不同。理论证明，两层神经网络可以无限逼近任意连续函数。

这是什么意思呢？也就是说，面对复杂的非线性分类任务，两层（带一个隐藏层）神经网络可以分类的很好。

下面就是一个例子（此两图来自colah的博客），红色的线与蓝色的线代表数据。而红色区域和蓝色区域代表由神经网络划开的区域，两者的分界线就是决策分界。

![](http://img.blog.csdn.net/20160114194436642 "神经网络")

```
                                      图11 两层神经网络（决策分界）
```

可以看到，这个两层神经网络的决策分界是非常平滑的曲线，而且分类的很好。有趣的是，前面已经学到过，单层网络只能做线性分类任务。而两层神经网络中的后一层也是线性分类层，应该只能做线性分类任务。为什么两个线性分类任务结合就可以做非线性分类任务？

我们可以把输出层的决策分界单独拿出来看一下。就是下图。

![](http://img.blog.csdn.net/20160114194458571 "神经网络")

```
                                      图12 两层神经网络（空间变换）
```

可以看到，输出层的决策分界仍然是直线。关键就是，从输入层到隐藏层时，数据发生了空间变换。也就是说，两层神经网络中，隐藏层对原始的数据进行了一个空间变换，使其可以被线性分类，然后输出层的决策分界划出了一个线性分类分界线，对其进行分类。

这样就导出了两层神经网络可以做非线性分类的关键–隐藏层。联想到我们一开始推导出的矩阵公式，我们知道，矩阵和向量相乘，本质上就是对向量的坐标空间进行一个变换。因此，隐藏层的参数矩阵的作用就是使得数据的原始坐标空间从线性不可分，转换成了线性可分。

两层神经网络通过两层的线性模型模拟了数据内真实的非线性函数。因此，多层的神经网络的本质就是复杂函数拟合。

下面来讨论一下隐藏层的节点数设计。在设计一个神经网络时，输入层的节点数需要与特征的维度匹配，输出层的节点数要与目标的维度匹配。而中间层的节点数，却是由设计者指定的。因此，“自由”把握在设计者的手中。但是，节点数设置的多少，却会影响到整个模型的效果。如何决定这个自由层的节点数呢？目前业界没有完善的理论来指导这个决策。一般是根据经验来设置。较好的方法就是预先设定几个可选值，通过切换这几个值来看整个模型的预测效果，选择效果最好的值作为最终选择。这种方法又叫做Grid Search（网格搜索）。

了解了两层神经网络的结构以后，我们就可以看懂其它类似的结构图。例如EasyPR字符识别网络[架构](http://lib.csdn.net/base/architecture)（下图）。

![](http://img.blog.csdn.net/20160114194529593 "神经网络")

```
                                      图13 EasyPR字符识别网络
```

EasyPR使用了字符的图像去进行字符文字的识别。输入是120维的向量。输出是要预测的文字类别，共有65类。根据实验，我们[测试](http://lib.csdn.net/base/softwaretest)了一些隐藏层数目，发现当值为40时，整个网络在测试集上的效果较好，因此选择网络的最终结构就是120，40，65。

**3.训练**

下面简单介绍一下两层神经网络的训练。

在Rosenblat提出的感知器模型中，模型中的参数可以被训练，但是使用的方法较为简单，并没有使用目前[机器学习](http://lib.csdn.net/base/machinelearning)中通用的方法，这导致其扩展性与适用性非常有限。从两层神经网络开始，神经网络的研究人员开始使用机器学习相关的技术进行神经网络的训练。例如用大量的数据（1000-10000左右），使用[算法](http://lib.csdn.net/base/datastructure)进行优化等等，从而使得模型训练可以获得性能与数据利用上的双重优势。

机器学习模型训练的目的，就是使得参数尽可能的与真实的模型逼近。具体做法是这样的。首先给所有参数赋上随机值。我们使用这些随机生成的参数值，来预测训练数据中的样本。样本的预测目标为yp，真实目标为y。那么，定义一个值loss，计算公式如下。

loss = \(yp - y\)2

这个值称之为损失（loss），我们的目标就是使对所有训练数据的损失和尽可能的小。

如果将先前的神经网络预测的矩阵公式带入到yp中（因为有z=yp），那么我们可以把损失写为关于参数（parameter）的函数，这个函数称之为损失函数（loss function）。下面的问题就是求：如何优化参数，能够让损失函数的值最小。

此时这个问题就被转化为一个优化问题。一个常用方法就是高等数学中的求导，但是这里的问题由于参数不止一个，求导后计算导数等于0的运算量很大，所以一般来说解决这个优化问题使用的是梯度下降算法。梯度下降算法每次计算参数在当前的梯度，然后让参数向着梯度的反方向前进一段距离，不断重复，直到梯度接近零时截止。一般这个时候，所有的参数恰好达到使损失函数达到一个最低值的状态。

在神经网络模型中，由于结构复杂，每次计算梯度的代价很大。因此还需要使用反向传播算法。反向传播算法是利用了神经网络的结构进行的计算。不一次计算所有参数的梯度，而是从后往前。首先计算输出层的梯度，然后是第二个参数矩阵的梯度，接着是中间层的梯度，再然后是第一个参数矩阵的梯度，最后是输入层的梯度。计算结束以后，所要的两个参数矩阵的梯度就都有了。

反向传播算法可以直观的理解为下图。梯度的计算从后往前，一层层反向传播。前缀E代表着相对导数的意思。

![](http://img.blog.csdn.net/20160114194607982 "神经网络")

```
                                         图14 反向传播算法
```

反向传播算法的启示是数学中的链式法则。在此需要说明的是，尽管早期神经网络的研究人员努力从生物学中得到启发，但从BP算法开始，研究者们更多地从数学上寻求问题的最优解。不再盲目模拟人脑网络是神经网络研究走向成熟的标志。正如科学家们可以从鸟类的飞行中得到启发，但没有必要一定要完全模拟鸟类的飞行方式，也能制造可以飞天的飞机。

优化问题只是训练中的一个部分。机器学习问题之所以称为学习问题，而不是优化问题，就是因为它不仅要求数据在训练集上求得一个较小的误差，在测试集上也要表现好。因为模型最终是要部署到没有见过训练数据的真实场景。提升模型在测试集上的预测效果的主题叫做泛化（generalization），相关方法被称作正则化（regularization）。神经网络中常用的泛化技术有权重衰减等。

## **三 多层神经网络（深度学习）** {#三-多层神经网络深度学习}

**1.结构**

我们延续两层神经网络的方式来设计一个多层神经网络。

在两层神经网络的输出层后面，继续添加层次。原来的输出层变成中间层，新加的层次成为新的输出层。所以可以得到下图。

![](http://img.blog.csdn.net/20160114194720415 "神经网络")

```
                                      图15 多层神经网络

依照这样的方式不断添加，我们可以得到更多层的多层神经网络。公式推导的话其实跟两层神经网络类似，使用矩阵运算的话就仅仅是加一个公式而已。
```

在已知输入a\(1\)，参数W\(1\)，W\(2\)，W\(3\)的情况下，输出z的推导公式如下：

g\(W\(1\) \* a\(1\)\) = a\(2\);

g\(W\(2\) \* a\(2\)\) = a\(3\);

g\(W\(3\) \* a\(3\)\) = z;

```
多层神经网络中，输出也是按照一层一层的方式来计算。从最外面的层开始，算出所有单元的值以后，再继续计算更深一层。只有当前层所有单元的值都计算完毕以后，才会算下一层。有点像计算向前不断推进的感觉。所以这个过程叫做“正向传播”。
```

下面讨论一下多层神经网络中的参数。

首先我们看第一张图，可以看出W\(1\)中有6个参数，W\(2\)中有4个参数，W\(3\)中有6个参数，所以整个神经网络中的参数有16个（这里我们不考虑偏置节点，下同）。

假设我们将中间层的节点数做一下调整。第一个中间层改为3个单元，第二个中间层改为4个单元。

经过调整以后，整个网络的参数变成了33个。

虽然层数保持不变，但是第二个神经网络的参数数量却是第一个神经网络的接近两倍之多，从而带来了更好的表示（represention）能力。表示能力是多层神经网络的一个重要性质，下面会做介绍。

在参数一致的情况下，我们也可以获得一个“更深”的网络。

![](http://img.blog.csdn.net/20160114194834758 "神经网络")

```
                                   图16 多层神经网络（更深的层次）
```

上图的网络中，虽然参数数量仍然是33，但却有4个中间层，是原来层数的接近两倍。这意味着一样的参数数量，可以用更深的层次去表达。

**2.效果**

与两层层神经网络不同。多层神经网络中的层数增加了很多。

增加更多的层次有什么好处？更深入的表示特征，以及更强的函数模拟能力。

更深入的表示特征可以这样理解，随着网络的层数增加，每一层对于前一层次的抽象表示更深入。在神经网络中，每一层神经元学习到的是前一层神经元值的更抽象的表示。例如第一个隐藏层学习到的是“边缘”的特征，第二个隐藏层学习到的是由“边缘”组成的“形状”的特征，第三个隐藏层学习到的是由“形状”组成的“图案”的特征，最后的隐藏层学习到的是由“图案”组成的“目标”的特征。通过抽取更抽象的特征来对事物进行区分，从而获得更好的区分与分类能力。

关于逐层特征学习的例子，可以参考下图。

![](http://img.blog.csdn.net/20160114194911190 "神经网络")

```
                                   图17 多层神经网络（特征学习）
```

更强的函数模拟能力是由于随着层数的增加，整个网络的参数就越多。而神经网络其实本质就是模拟特征与目标之间的真实关系函数的方法，更多的参数意味着其模拟的函数可以更加的复杂，可以有更多的容量（capcity）去拟合真正的关系。

通过研究发现，在参数数量一样的情况下，更深的网络往往具有比浅层的网络更好的识别效率。这点也在ImageNet的多次大赛中得到了证实。从2012年起，每年获得ImageNet冠军的深度神经网络的层数逐年增加，2015年最好的方法GoogleNet是一个多达22层的神经网络。

在最新一届的ImageNet大赛上，目前拿到最好成绩的MSRA团队的方法使用的更是一个深达152层的网络！关于这个方法更多的信息有兴趣的可以查阅ImageNet网站。

**3.训练**

在单层神经网络时，我们使用的激活函数是sgn函数。到了两层神经网络时，我们使用的最多的是sigmoid函数。而到了多层神经网络时，通过一系列的研究发现，ReLU函数在训练多层神经网络时，更容易收敛，并且预测性能更好。因此，目前在[深度学习](http://lib.csdn.net/base/deeplearning)中，最流行的非线性函数是ReLU函数。ReLU函数不是传统的非线性函数，而是分段线性函数。其表达式非常简单，就是y=max\(x,0\)。简而言之，在x大于0，输出就是输入，而在x小于0时，输出就保持为0。这种函数的设计启发来自于生物神经元对于激励的线性响应，以及当低于某个阈值后就不再响应的模拟。

在多层神经网络中，训练的主题仍然是优化和泛化。当使用足够强的计算芯片（例如GPU图形加速卡）时，梯度下降算法以及反向传播算法在多层神经网络中的训练中仍然工作的很好。目前学术界主要的研究既在于开发新的算法，也在于对这两个算法进行不断的优化，例如，增加了一种带动量因子（momentum）的梯度下降算法。

在深度学习中，泛化技术变的比以往更加的重要。这主要是因为神经网络的层数增加了，参数也增加了，表示能力大幅度增强，很容易出现过拟合现象。因此正则化技术就显得十分重要。目前，Dropout技术，以及数据扩容（Data-Augmentation）技术是目前使用的最多的正则化技术。

## **四 深度学习必知的框架** {#四-深度学习必知的框架}

GitHub上其实还有很多不错的开源项目值得关注，首先我们推荐目前规模人气最高的TOP3：

**一、Caffe**。源自加州伯克利分校的Caffe被广泛应用，包括Pinterest这样的web大户。与TensorFlow一样，Caffe也是由C++开发，Caffe也是Google今年早些时候发布的DeepDream项目（可以识别喵星人的人工[智能](http://lib.csdn.net/base/aiplanning)神经网络）的基础。

**二、Theano**。2008年诞生于蒙特利尔理工学院，Theano派生出了大量深度学习[Python](http://lib.csdn.net/base/python)软件包，最著名的包括Blocks和Keras。

**三、Torch**。Torch诞生已经有十年之久，但是真正起势得益于去年Facebook开源了大量Torch的深度学习模块和扩展。Torch另外一个特殊之处是采用了不怎么流行的编程语言Lua（该语言曾被用来开发视频游戏）。  
除了以上三个比较成熟知名的项目，还有很多有特色的深度学习开源框架也值得关注：

四、Brainstorm。来自瑞士[人工智能](http://lib.csdn.net/base/aiframework)实验室IDSIA的一个非常发展前景很不错的深度学习软件包，Brainstorm能够处理上百层的超级深度神经网络——所谓的公路网络Highway Networks。

五、Chainer。来自一个日本的深度学习创业公司Preferred Networks，今年6月发布的一个Python框架。Chainer的设计基于define by run原则，也就是说，该网络在运行中动态定义，而不是在启动时定义，这里有Chainer的详细文档。

六、Deeplearning4j。 顾名思义，Deeplearning4j是”for[Java](http://lib.csdn.net/base/javase)”的深度学习框架，也是首个商用级别的深度学习开源库。Deeplearning4j由创业公司Skymind于2014年6月发布，使用 Deeplearning4j的不乏埃森哲、雪弗兰、博斯咨询和IBM等明星企业。

DeepLearning4j是一个面向生产环境和商业应用的高成熟度深度学习开源库，可与[Hadoop](http://lib.csdn.net/base/hadoop)和[Spark](http://lib.csdn.net/base/spark)集成，即插即用，方便开发者在APP中快速集成深度学习功能，可应用于以下深度学习领域：  
   人脸/图像识别  
   语音搜索  
   语音转文字（Speech to text）  
   垃圾信息过滤（异常侦测）  
   电商欺诈侦测  
七、Marvin。是普林斯顿大学视觉工作组新推出的C++框架。该团队还提供了一个文件用于将Caffe模型转化成语Marvin兼容的模式。

八、ConvNetJS。这是斯坦福大学博士生Andrej Karpathy开发浏览器插件，基于万能的[JavaScript](http://lib.csdn.net/base/javascript)可以在你的游览器中训练神经网络。Karpathy还写了一个ConvNetJS的入门教程，以及一个简洁的浏览器演示项目。

九、MXNet。出自CXXNet、Minerva、Purine等项目的开发者之手，主要用C++编写。MXNet强调提高内存使用的效率，甚至能在智能手机上运行诸如图像识别等任务。

十、Neon。由创业公司Nervana Systems于今年五月开源，在某些基准测试中，由Python和Sass开发的Neon的测试成绩甚至要优于Caffeine、Torch和谷歌的TensorFlow。

# 

# 深度学习领域的学术研究可以包含四部分：

优化（Optimization），泛化（Generalization），表达（Representation）以及应（Applications）。除了应用（Applications）之外每个部分又可以分成实践和理论两个方面。

---

**优化（Optimization）**：深度学习的问题最后似乎总能变成优化问题，这个时候数值优化的方法就变得尤其重要。

从实践方面来说，现在最为推崇的方法依旧是随机梯度递减，这样一个极其简单的方法以其强悍的稳定性深受广大研究者的喜爱，而不同的人还会结合动量（momentum）、伪牛顿方法（Pseudo-Newton）以及自动步长等各种技巧。此外，深度学习模型优化过程的并行化也是一个非常热的点，近年在分布式系统的会议上相关论文也逐渐增多。

在理论方面，目前研究的比较清楚的还是凸优化（Convex Optimization），而对于非凸问题的理论还严重空缺，然而深度学习大多数有效的方法都是非凸的。现在有一些对深度学习常用模型及其目标函数的特性研究，期待能够发现非凸问题中局部最优解的相关规律。

**泛化（Generalization）**：一个模型的泛化能力是指它在训练数据集上的误差是否能够接近所有可能[测试](http://lib.csdn.net/base/softwaretest)数据误差的均值。泛化误差大致可以理解成测试数据集误差和训练数据集误差之差。在深度学习领域变流行之前，如何控制泛化误差一直是[机器学习](http://lib.csdn.net/base/machinelearning)领域的主流问题。

从实践方面来说，之前许多人担心的深度神经网络泛化能力较差的问题，在现实使用中并没有表现得很明显。这一方面源于[大数据](http://lib.csdn.net/base/hadoop)时代样本巨大的数量，另一方面近年出现了一些新的在实践上比较有效的控制泛化误差（Regularization）的方法，比如Dropout和DropConnect，以及非常有效的数据扩增（Data Agumentation）技术。是否还有其它实践中会比较有效的泛化误差控制方法一直是研究者们的好奇点，比如是否可以通过博弈法避免过拟合，以及是否可以利用无标记（Unlabeled）样本来辅助泛化误差的控制。

从理论方面来说，深度学习的有效性使得PAC学习（Probably Approximately Correct Learning）相关的理论倍受质疑。这些理论无一例外地属于“上界的上界”的一个证明过程，而其本质无外乎各种集中不等式（Concentration Inequality）和复杂性度量（Complexity Measurement）的变种，因此它对深度学习模型有相当不切实际的估计。这不应该是泛函理论已经较为发达的当下出现的状况，因此下一步如何能够从理论上分析深度学习模型的泛化能力也会是一个有趣的问题。而这个研究可能还会牵涉表达（Representation，见下）的一些理论。

**表达（Representation）**：这方面主要指的是深度学习模型和它要解决的问题之间的关系，比如给出一个设计好的深度学习模型，它适合表达什么样的问题，以及给定一个问题是否存在一个可以进行表达的深度学习模型。

这方面的实践主要是两个主流，一方面那些笃信无监督学习（Unsupervised Learning）可行性的研究者们一直在寻找更好的无监督学习目标及其评价方法，以使得机器能够自主进行表达学习变得可能。这实际上包括了受限波尔兹曼模型（Restricted Boltzmann Machine），稀疏编码（Sparse Coding）和自编码器（Auto-encoder）等。另一方面，面对实际问题的科学家们一直在凭借直觉设计深度学习模型的结构来解决这些问题。这方面出现了许多成功的例子，比如用于视觉和[语音识别](http://lib.csdn.net/base/vras)的卷积神经网络（Convolutional Neural Network），以及能够进行自我演绎的深度回归神经网络（Recurrent Neural Network）和会自主玩游戏的深度强化学习（Reinforcement Learning）模型。绝大多数的深度学习研究者都集中在这方面，而这些也恰恰能够带来最大的学术影响力。

然而，有关表达（Representation）的理论，除了从认知心理学和神经科学借用的一些启发之外，几乎是空白。这主要是因为是否能够存在表达的理论实际上依赖于具体的问题，而面对具体问题的时候目前唯一能做的事情就是去类比现实存在的[智能](http://lib.csdn.net/base/aiplanning)体（人类）是如何解决这一问题的，并设计模型来将它归约为学习[算法](http://lib.csdn.net/base/datastructure)。我直觉上认为，终极的表达理论就像是拉普拉斯幽灵（Laplace’s Demon）一样，如果存在它便无所不知，也因此它的存在会产生矛盾，使得这一理论实际上只能无限逼近。

**应用（Applications）**：深度学习的发展伴随着它对其它领域的革命过程。在过去的数年中，深度学习的应用能力几乎是一种“敢想就能成”的状态。这当然得益于现今各行各业丰富的数据集以及计算机计算能力的提升，同时也要归功于过去近三十年的领域经验。未来，深度学习将继续解决各种识别（Recognition）相关的问题，比如视觉（图像分类、分割，计算摄影学），语音（语音识别），[自然语言](http://lib.csdn.net/base/nlp)（文本理解）；同时，在能够演绎（Ability to Act）的方面如图像文字描述、语音合成、自动翻译、段落总结等也会逐渐出现突破，更可能协助寻找NP难（NP-Hard）问题在限定输入集之后的可行算法。所有的这些都可能是非常好的研究点，能够带来经济和学术双重的利益。

# 深度学习数据集

_\*_先来个不能错过的数据集网站（[深度学习](http://lib.csdn.net/base/deeplearning)者的福音）：\*  
[http://deeplearning.net/datasets/](http://deeplearning.net/datasets/)\*\*

首先说说几个收集数据集的网站：  
1、Public Data Sets on Amazon Web Services \(AWS\)  
[http://aws.amazon.com/datasets](http://aws.amazon.com/datasets)  
Amazon从2008年开始就为开发者提供几十TB的开发数据。

2、Yahoo! Webscope  
[http://webscope.sandbox.yahoo.com/index.php](http://webscope.sandbox.yahoo.com/index.php)

3、Konect is a collection of network datasets  
[http://konect.uni-koblenz.de/](http://konect.uni-koblenz.de/)

4、Stanford Large Network Dataset Collection  
[http://snap.stanford.edu/data/index.html](http://snap.stanford.edu/data/index.html)

再就是说说几个跟互联网有关的数据集：  
1、Dataset for “Statistics and Social Network of YouTube Videos”  
[http://netsg.cs.sfu.ca/youtubedata/](http://netsg.cs.sfu.ca/youtubedata/)

2、1998 World Cup Web Site Access Logs  
[http://ita.ee.lbl.gov/html/contrib/WorldCup.html](http://ita.ee.lbl.gov/html/contrib/WorldCup.html)  
这个是1998年世界杯期间的数据集。从1998/04/26 到 1998/07/26 的92天中，发生了 1,352,804,107次请求。

3、Page view statistics for Wikimedia projects  
[http://dammit.lt/wikistats/](http://dammit.lt/wikistats/)

4、AOL Search Query Logs - RP  
[http://www.researchpipeline.com/mediawiki/index.php?title=AOL\_Search\_Query\_Logs](http://www.researchpipeline.com/mediawiki/index.php?title=AOL_Search_Query_Logs)

5、livedoor gourmet  
[http://blog.livedoor.jp/techblog/archives/65836960.html](http://blog.livedoor.jp/techblog/archives/65836960.html)

海量图像数据集：  
1、ImageNet  
[http://www.image-net.org/](http://www.image-net.org/)  
包含1400万的图像。

2、Tiny Images Dataset  
[http://horatio.cs.nyu.edu/mit/tiny/data/index.html](http://horatio.cs.nyu.edu/mit/tiny/data/index.html)  
包含8000万的32x32图像。

3、 MirFlickr1M  
[http://press.liacs.nl/mirflickr/](http://press.liacs.nl/mirflickr/)  
Flickr中的100万的图像集。

4、 CoPhIR  
[http://cophir.isti.cnr.it/whatis.html](http://cophir.isti.cnr.it/whatis.html)  
Flickr中的1亿600万的图像

5、SBU captioned photo dataset  
[http://dsl1.cewit.stonybrook.edu/~vicente/sbucaptions/](http://dsl1.cewit.stonybrook.edu/~vicente/sbucaptions/)  
Flickr中的100万的图像集。

6、Large-Scale Image Annotation using Visual Synset\(ICCV 2011\)  
[http://cpl.cc.gatech.edu/projects/VisualSynset/](http://cpl.cc.gatech.edu/projects/VisualSynset/)  
包含2亿图像

7、NUS-WIDE  
[http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm](http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm)  
Flickr中的27万的图像集。

8、SUN dataset  
[http://people.csail.mit.edu/jxiao/SUN/](http://people.csail.mit.edu/jxiao/SUN/)  
包含13万的图像

9、MSRA-MM  
[http://research.microsoft.com/en-us/projects/msrammdata/](http://research.microsoft.com/en-us/projects/msrammdata/)  
包含100万的图像，23000视频

10、TRECVID  
[http://trecvid.nist.gov/](http://trecvid.nist.gov/)

截止目前好像还没有国内的企业或者组织开放自己的数据集。希望也能有企业开发自己的数据集给研究人员使用，从而推动海量数据处理在国内的发展！

2014/07/07 雅虎发布超大Flickr数据集 1亿的图片+视频  
[http://yahoolabs.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images-for](http://yahoolabs.tumblr.com/post/89783581601/one-hundred-million-creative-commons-flickr-images-for)

100多个有趣的数据集  
[http://www.csdn.net/article/2014-06-06/2820111-100-Interesting-Data-Sets-for-Statistics](http://www.csdn.net/article/2014-06-06/2820111-100-Interesting-Data-Sets-for-Statistics)

## 视频人体姿态数据集 {#视频人体姿态数据集}

**1. Weizmann 人体行为库**

此[数据库](http://lib.csdn.net/base/mysql)一共包括90段视频，这些视频分别是由9个人执行了10个不同的动作（bend, jack, jump, pjump, run, side, skip, walk, wave1,wave2）。视频的背景，视角以及摄像头都是静止的。而且该数据库提供标注好的前景轮廓视频。不过此数据库的正确率已经达到100%了。下载地址：[http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html)  
![](http://img.blog.csdn.net/20151120145237333 "这里写图片描述")  
**2. KTH人体行为数据库**  
该数据库包括6类行为（walking, jogging, running, boxing, hand waving, hand clapping）,是由25个不同的人执行的，分别在四个场景下，一共有599段视频。背景相对静止，除了镜头的拉近拉远，摄像机的运动比较轻微。这个数据库是现在的benchmark，正确率需要达到95.5%以上才能够发文章。下载地址：[http://www.nada.kth.se/cvap/actions/](http://www.nada.kth.se/cvap/actions/)  
![](http://img.blog.csdn.net/20151120145338437 "这里写图片描述")  
**3. INRIA XMAX多视角视频库**  
该数据库从五个视角获得，一共11个人执行14种行为。室内四个方向和头顶一共安装5个摄像头。另外背景和光照基本不变。下载地址：[http://4drepository.inrialpes.fr/public/viewgroup/6](http://4drepository.inrialpes.fr/public/viewgroup/6)  
![](http://img.blog.csdn.net/20151120145449665 "这里写图片描述")  
**4. UCF Sports 数据库**  
该视频包括150段关于体育的视频，一共有13个动作。实验室采用留一交叉验证法。2011年cvpr有几篇都用这个数据库，正确率要达到87%才能发文章。下载地址：[http://vision.eecs.ucf.edu/data.html](http://vision.eecs.ucf.edu/data.html)  
**6. Olympic sports dataset**

该数据库有16种行为，783段视频。现在的正确率大约在75%左右。下载地址：[http://vision.stanford.edu/Datasets/OlympicSports/](http://vision.stanford.edu/Datasets/OlympicSports/)

## UCI收集的机器学习数据集 {#uci收集的机器学习数据集}

[ftp://pami.sjtu.edu.cn](ftp://pami.sjtu.edu.cn/)  
[http://www.ics.uci.edu/~mlearn/](http://www.ics.uci.edu/~mlearn/)\MLRepository.htm  
**样本数据库**  
[http://kdd.ics.uci.edu/](http://kdd.ics.uci.edu/)  
[http://www.ics.uci.edu/~mlearn/MLRepository.html](http://www.ics.uci.edu/~mlearn/MLRepository.html)

## CASIA WebFace Database {#casia-webface-database}

中科院自动化研究所的几种数据集，里面包含掌纹，手写体，人体动作等6种数据集；需要按照说明申请，免费使用。  
[http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)

![](http://img.blog.csdn.net/20151120150109648 "这里写图片描述")

## 微软人体姿态数据库 MSRC-12 Gesture Dataset {#微软人体姿态数据库-msrc-12-gesture-dataset}

手势数据集  
[http://www.datatang.com/data/46521](http://www.datatang.com/data/46521)

备注：数据堂链接：[http://www.datatang.com/](http://www.datatang.com/)  
![](http://img.blog.csdn.net/20151120150505049 "这里写图片描述")

## 文本分类数据集 {#文本分类数据集}

一个数据集是可以用的，即rainbow的数据集  
[http://www-2.cs.cmu.edu/afs/cs/p](http://www-2.cs.cmu.edu/afs/cs/p)… ww/naive-bayes.html

## 其余杂数据集 {#其余杂数据集}

癌症基因：  
[http://www.broad.mit.edu/cgi-bin/cancer/datasets.cgi](http://www.broad.mit.edu/cgi-bin/cancer/datasets.cgi)  
金融数据：  
[http://lisp.vse.cz/pkdd99/Challenge/chall.htm](http://lisp.vse.cz/pkdd99/Challenge/chall.htm)  
各种数据集：  
[http://kdd.ics.uci.edu/summary.data.type.html](http://kdd.ics.uci.edu/summary.data.type.html)  
[http://www.mlnet.org/cgi-bin/mlnetois.pl/?File=datasets.html](http://www.mlnet.org/cgi-bin/mlnetois.pl/?File=datasets.html)  
[http://lib.stat.cmu.edu/datasets/](http://lib.stat.cmu.edu/datasets/)  
[http://dctc.sjtu.edu.cn/adaptive/datasets/](http://dctc.sjtu.edu.cn/adaptive/datasets/)  
[http://fimi.cs.helsinki.fi/data/](http://fimi.cs.helsinki.fi/data/)  
[http://www.almaden.ibm.com/software/quest/Resources/index.shtml](http://www.almaden.ibm.com/software/quest/Resources/index.shtml)  
[http://miles.cnuce.cnr.it/~palmeri/datam/DCI/](http://miles.cnuce.cnr.it/~palmeri/datam/DCI/)

