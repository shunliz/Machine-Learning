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

