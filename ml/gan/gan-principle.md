# GAN原理

---

![](https://pic4.zhimg.com/v2-95c709a87749e0778248fc8fdd289b83_r.jpg)

Generative Adversarial Network，就是大家耳熟能详的GAN，由Ian Goodfellow首先提出，在这两年更是深度学习中最热门的东西，仿佛什么东西都能由GAN做出来。我最近刚入门GAN，看了些资料，做一些笔记。

## 1.Generation

什么是生成（generation）？就是模型通过学习一些数据，然后生成类似的数据。让机器看一些动物图片，然后自己来产生动物的图片，这就是生成。

以前就有很多可以用来生成的技术了，比如auto-encoder（自编码器），结构如下图：

![](https://pic3.zhimg.com/v2-6286121d0904483dfef269f8093eb842_b.png)

你训练一个encoder，把input转换成code，然后训练一个decoder，把code转换成一个image，然后计算得到的image和input之间的MSE（mean square error），训练完这个model之后，取出后半部分NN Decoder，输入一个随机的code，就能generate一个image。

但是auto-encoder生成image的效果，当然看着很别扭啦，一眼就能看出真假。所以后来还提出了比如VAE这样的生成模型，我对此也不是很了解，在这就不细说。

上述的这些生成模型，其实有一个非常严重的弊端。比如VAE，它生成的image是希望和input越相似越好，但是model是如何来衡量这个相似呢？model会计算一个loss，采用的大多是MSE，即每一个像素上的均方差。loss小真的表示相似嘛？![](https://pic2.zhimg.com/v2-ff75e66a8b9042bcde8fbd69f570e045_b.png)

比如这两张图，第一张，我们认为是好的生成图片，第二张是差的生成图片，但是对于上述的model来说，这两张图片计算出来的loss是一样大的，所以会认为是一样好的图片。

这就是上述生成模型的弊端，用来衡量生成图片好坏的标准并不能很好的完成想要实现的目的。于是就有了下面要讲的GAN。

  


## 2.GAN

大名鼎鼎的GAN是如何生成图片的呢？首先大家都知道GAN有两个网络，一个是generator，一个是discriminator，从二人零和博弈中受启发，通过两个网络互相对抗来达到最好的生成效果。流程如下：

![](https://pic2.zhimg.com/v2-6277da1cacd7a7fb7c0d326eb47c2135_b.png)

主要流程类似上面这个图。首先，有一个一代的generator，它能生成一些很差的图片，然后有一个一代的discriminator，它能准确的把生成的图片，和真实的图片分类，简而言之，这个discriminator就是一个二分类器，对生成的图片输出0，对真实的图片输出1。

接着，开始训练出二代的generator，它能生成稍好一点的图片，能够让一代的discriminator认为这些生成的图片是真实的图片。然后会训练出一个二代的discriminator，它能准确的识别出真实的图片，和二代generator生成的图片。以此类推，会有三代，四代。。。n代的generator和discriminator，最后discriminator无法分辨生成的图片和真实图片，这个网络就拟合了。

这就是GAN，运行过程就是这么的简单。这就结束了嘛？显然没有，下面还要介绍一下GAN的原理。

  


## 3.原理

首先我们知道真实图片集的分布![](http://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29 "P\_{data}\(x\)")，x是一个真实图片，可以想象成一个向量，这个向量集合的分布就是![](http://www.zhihu.com/equation?tex=P_%7Bdata%7D "P\_{data}")。我们需要生成一些也在这个分布内的图片，如果直接就是这个分布的话，怕是做不到的。

我们现在有的generator生成的分布可以假设为![](http://www.zhihu.com/equation?tex=P_G%28x%3B%5Ctheta%29 "P\_G\(x;\theta\)")，这是一个由![](http://www.zhihu.com/equation?tex=%5Ctheta "\theta")控制的分布，![](http://www.zhihu.com/equation?tex=%5Ctheta "\theta")是这个分布的参数（如果是高斯混合模型，那么![](http://www.zhihu.com/equation?tex=%5Ctheta "\theta")就是每个高斯分布的平均值和方差）

假设我们在真实分布中取出一些数据，![](http://www.zhihu.com/equation?tex=%5C%7Bx%5E1%2C+x%5E2%2C+%5Cdots%2Cx%5Em+%5C%7D "\{x^1, x^2, \dots,x^m \}")，我们想要计算一个似然![](http://www.zhihu.com/equation?tex=P_G%28x%5Ei%3B%5Ctheta%29 "P\_G\(x^i;\theta\)")

对于这些数据，在生成模型中的似然就是![](http://www.zhihu.com/equation?tex=L+%3D+%5Cprod_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x%5Ei%3B%5Ctheta%29+ "L = \prod\_{i=1}^{m}P\_G\(x^i;\theta\) ")

我们想要最大化这个似然，等价于让generator生成那些真实图片的概率最大。这就变成了一个最大似然估计的问题了，我们需要找到一个![](http://www.zhihu.com/equation?tex=%5Ctheta+%5E%2A "\theta ^\*")来最大化这个似然。

![](http://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D%0A%5Ctheta+%5E%2A+%26%3D+arg%5C+%5Cmax_%7B%5Ctheta%7D%5Cprod_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x%5Ei%3B%5Ctheta%29+%5C%5C%0A%26%3Darg%5C+%5Cmax_%7B%5Ctheta%7D%5C+log%5Cprod_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x%5Ei%3B%5Ctheta%29+%5C%5C%0A%26%3Darg%5C+%5Cmax_%7B%5Ctheta%7D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7DlogP_G%28x%5Ei%3B%5Ctheta%29+%5C%5C%0A%26+%5Capprox+arg%5C+%5Cmax_%7B%5Ctheta%7D%5C+E_%7Bx%5Csim+P_%7Bdata%7D%7D%5BlogP_G%28x%3B%5Ctheta%29%5D+%5C%5C%0A%26+%3D+arg%5C+%5Cmax_%7B%5Ctheta%7D%5Cint_%7Bx%7D+P_%7Bdata%7D%28x%29logP_G%28x%3B%5Ctheta%29dx+-+%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29logP_%7Bdata%7D%28x%29dx+%5C%5C%0A%26%3Darg%5C+%5Cmax_%7B%5Ctheta%7D%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29%28logP_G%28x%3B%5Ctheta%29-logP_%7Bdata%7D%28x%29%29dx+%5C%5C%0A%26%3Darg%5C+%5Cmin_%7B%5Ctheta%7D%5Cint_%7Bx%7DP_%7Bdata%7D%28x%29log+%5Cfrac%7BP_%7Bdata%7D%28x%29%7D%7BP_G%28x%3B%5Ctheta%29%7Ddx+%5C%5C%0A%26%3Darg%5C+%5Cmin_%7B%5Ctheta%7D%5C+KL%28P_%7Bdata%7D%28x%29%7C%7CP_G%28x%3B%5Ctheta%29%29%0A%5Cend%7Balign%7D%0A+ "\begin{align}
\theta ^\* &amp;= arg\ \max\_{\theta}\prod\_{i=1}^{m}P\_G\(x^i;\theta\) \\
&amp;=arg\ \max\_{\theta}\ log\prod\_{i=1}^{m}P\_G\(x^i;\theta\) \\
&amp;=arg\ \max\_{\theta} \sum\_{i=1}^{m}logP\_G\(x^i;\theta\) \\
&amp; \approx arg\ \max\_{\theta}\ E\_{x\sim P\_{data}}\[logP\_G\(x;\theta\)\] \\
&amp; = arg\ \max\_{\theta}\int\_{x} P\_{data}\(x\)logP\_G\(x;\theta\)dx - \int\_{x}P\_{data}\(x\)logP\_{data}\(x\)dx \\
&amp;=arg\ \max\_{\theta}\int\_{x}P\_{data}\(x\)\(logP\_G\(x;\theta\)-logP\_{data}\(x\)\)dx \\
&amp;=arg\ \min\_{\theta}\int\_{x}P\_{data}\(x\)log \frac{P\_{data}\(x\)}{P\_G\(x;\theta\)}dx \\
&amp;=arg\ \min\_{\theta}\ KL\(P\_{data}\(x\)\|\|P\_G\(x;\theta\)\)
\end{align}
 ")

  


寻找一个![](http://www.zhihu.com/equation?tex=%5Ctheta+%5E%2A "\theta ^\*")来最大化这个似然，等价于最大化log似然。因为此时这m个数据，是从真实分布中取的，所以也就约等于，真实分布中的所有x在![](http://www.zhihu.com/equation?tex=P_%7BG%7D "P\_{G}")分布中的log似然的期望。

真实分布中的所有x的期望，等价于求概率积分，所以可以转化成积分运算，因为减号后面的项和![](http://www.zhihu.com/equation?tex=%5Ctheta "\theta")无关，所以添上之后还是等价的。然后提出共有的项，括号内的反转，max变min，就可以转化为KL divergence的形式了，KL divergence描述的是两个概率分布之间的差异。

所以最大化似然，让generator最大概率的生成真实图片，也就是要找一个![](http://www.zhihu.com/equation?tex=%5Ctheta "\theta")让![](http://www.zhihu.com/equation?tex=P_G "P\_G")更接近于![](http://www.zhihu.com/equation?tex=P_%7Bdata%7D "P\_{data}")

那如何来找这个最合理的![](http://www.zhihu.com/equation?tex=%5Ctheta "\theta")呢？我们可以假设![](http://www.zhihu.com/equation?tex=P_G%28x%3B%5Ctheta%29 "P\_G\(x;\theta\)")是一个神经网络。

首先随机一个向量z，通过G\(z\)=x这个网络，生成图片x，那么我们如何比较两个分布是否相似呢？只要我们取一组sample z，这组z符合一个分布，那么通过网络就可以生成另一个分布![](http://www.zhihu.com/equation?tex=P_G "P\_G")，然后来比较与真实分布![](http://www.zhihu.com/equation?tex=P_%7Bdata%7D "P\_{data}")

大家都知道，神经网络只要有非线性激活函数，就可以去拟合任意的函数，那么分布也是一样，所以可以用一直正态分布，或者高斯分布，取样去训练一个神经网络，学习到一个很复杂的分布。

![](https://pic1.zhimg.com/v2-bc64f778f95312aa0c37d2ddb62358ec_b.png)

如何来找到更接近的分布，这就是GAN的贡献了。先给出GAN的公式：

![](http://www.zhihu.com/equation?tex=V%28G%2CD%29%3DE_%7Bx%5Csim+P_%7Bdata%7D%7D%5BlogD%28x%29%5D+%2B+E_%7Bx%5Csim+P_G%7D%5Blog%281-D%28x%29%29%5D "V\(G,D\)=E\_{x\sim P\_{data}}\[logD\(x\)\] + E\_{x\sim P\_G}\[log\(1-D\(x\)\)\]")

  


这个式子的好处在于，固定G，![](http://www.zhihu.com/equation?tex=%5Cmax%5C++V%28G%2CD%29 "\max\  V\(G,D\)")就表示![](http://www.zhihu.com/equation?tex=P_G "P\_G")和![](http://www.zhihu.com/equation?tex=P_%7Bdata%7D "P\_{data}")之间的差异，然后要找一个最好的G，让这个最大值最小，也就是两个分布之间的差异最小。

![](http://www.zhihu.com/equation?tex=G%5E%2A%3Darg%5C+%5Cmin_%7BG%7D%5C+%5Cmax_D%5C+V%28G%2CD%29 "G^\*=arg\ \min\_{G}\ \max\_D\ V\(G,D\)")

  


表面上看这个的意思是，D要让这个式子尽可能的大，也就是对于x是真实分布中，D\(x\)要接近与1，对于x来自于生成的分布，D\(x\)要接近于0，然后G要让式子尽可能的小，让来自于生成分布中的x，D\(x\)尽可能的接近1

现在我们先固定G，来求解最优的D

![](https://pic1.zhimg.com/v2-bc7aef1f4608f037f0ec08a88d8c71bc_b.png)![](https://pic2.zhimg.com/v2-0e6fdaf7666cfab881c59d2bee203671_b.png)对于一个给定的x，得到最优的D如上图，范围在\(0,1\)内，把最优的D带入![](http://www.zhihu.com/equation?tex=%5Cmax_D%5C+V%28G%2CD%29 "\max\_D\ V\(G,D\)")，可以得到：![](https://pic1.zhimg.com/v2-9cb3d142f47715df12378f105c11d1f4_b.png)

![](https://pic2.zhimg.com/v2-6bbb3dd20f5b01f864fc72481159a95d_b.png)JS divergence是KL divergence的对称平滑版本，表示了两个分布之间的差异，这个推导就表明了上面所说的，固定G，![](http://www.zhihu.com/equation?tex=%5Cmax_D%5C+V%28G%2CD%29 "\max\_D\ V\(G,D\)")表示两个分布之间的差异，最小值是-2log2，最大值为0。

现在我们需要找个G，来最小化![](http://www.zhihu.com/equation?tex=%5Cmax_D%5C+V%28G%2CD%29 "\max\_D\ V\(G,D\)")，观察上式，当![](http://www.zhihu.com/equation?tex=P_G%28x%29%3DP_%7Bdata%7D%28x%29 "P\_G\(x\)=P\_{data}\(x\)")时，G是最优的。

  


## 4.训练

有了上面推导的基础之后，我们就可以开始训练GAN了。结合我们开头说的，两个网络交替训练，我们可以在起初有一个![](http://www.zhihu.com/equation?tex=G_0 "G\_0")和![](http://www.zhihu.com/equation?tex=D_0 "D\_0")，先训练![](http://www.zhihu.com/equation?tex=D_0 "D\_0")找到![](http://www.zhihu.com/equation?tex=%5Cmax_D%5C+V%28G_0%2CD_0%29 "\max\_D\ V\(G\_0,D\_0\)")，然后固定![](http://www.zhihu.com/equation?tex=D_0 "D\_0")开始训练![](http://www.zhihu.com/equation?tex=G_0 "G\_0"),训练的过程都可以使用gradient descent，以此类推，训练![](http://www.zhihu.com/equation?tex=D_1%2CG_1%2CD_2%2CG_2%2C%5Cdots "D\_1,G\_1,D\_2,G\_2,\dots")

但是这里有个问题就是，你可能在![](http://www.zhihu.com/equation?tex=D_0%5E%2A "D\_0^\*")的位置取到了![](http://www.zhihu.com/equation?tex=%5Cmax_D%5C+V%28G_0%2CD_0%29%3DV%28G_0%2CD_0%5E%2A%29 "\max\_D\ V\(G\_0,D\_0\)=V\(G\_0,D\_0^\*\)")，然后更新![](http://www.zhihu.com/equation?tex=G_0 "G\_0")为![](http://www.zhihu.com/equation?tex=G_1 "G\_1"),可能![](http://www.zhihu.com/equation?tex=V%28G_1%2CD_0%5E%2A%29%3CV%28G_0%2CD_0%5E%2A%29 "V\(G\_1,D\_0^\*\)&amp;lt;V\(G\_0,D\_0^\*\)")了，但是并不保证会出现一个新的点![](http://www.zhihu.com/equation?tex=D_1%5E%2A "D\_1^\*")使得![](http://www.zhihu.com/equation?tex=V%28G_1%2CD_1%5E%2A%29+%3E+V%28G_0%2CD_0%5E%2A%29 "V\(G\_1,D\_1^\*\)&amp;gt; V\(G\_0,D\_0^\*\)")，这样更新G就没达到它原来应该要的效果，如下图所示：

![](https://pic2.zhimg.com/v2-f0cd8b79a1dc8fb08c9a0bd2b7424065_b.png)

避免上述情况的方法就是更新G的时候，不要更新G太多。

知道了网络的训练顺序，我们还需要设定两个loss function，一个是D的loss，一个是G的loss。下面是整个GAN的训练具体步骤：

![](https://pic3.zhimg.com/v2-9b9cf73c1064a87a26ef3c0f9eac7b76_b.png)

上述步骤在机器学习和深度学习中也是非常常见，易于理解。

  


## 5.存在的问题

但是上面G的loss function还是有一点小问题，下图是两个函数的图像：

![](https://pic4.zhimg.com/v2-e0ab404f7b693a4127ef887a3ffa2ba3_b.png)![](http://www.zhihu.com/equation?tex=log%281-D%28x%29%29 "log\(1-D\(x\)\)")是我们计算时G的loss function，但是我们发现，在D\(x\)接近于0的时候，这个函数十分平滑，梯度非常的小。这就会导致，在训练的初期，G想要骗过D，变化十分的缓慢，而上面的函数，趋势和下面的是一样的，都是递减的。但是它的优势是在D\(x\)接近0的时候，梯度很大，有利于训练，在D\(x\)越来越大之后，梯度减小，这也很符合实际，在初期应该训练速度更快，到后期速度减慢。  


所以我们把G的loss function修改为![](http://www.zhihu.com/equation?tex=minimize%5C+V+%3D+-%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dlog%28D%28x%5Ei%29%29 "minimize\ V = -\frac{1}{m}\sum\_{i=1}^{m}log\(D\(x^i\)\)")，这样可以提高训练的速度。

还有一个问题，在其他paper中提出，就是经过实验发现，经过许多次训练，loss一直都是平的，也就是![](http://www.zhihu.com/equation?tex=%5Cmax_D%5C+V%28G%2CD%29%3D0 "\max\_D\ V\(G,D\)=0")，JS divergence一直都是log2，![](http://www.zhihu.com/equation?tex=P_G "P\_G")和![](http://www.zhihu.com/equation?tex=P_%7Bdata%7D "P\_{data}")完全没有交集，但是实际上两个分布是有交集的，造成这个的原因是因为，我们无法真正计算期望和积分，只能使用sample的方法，如果训练的过拟合了，D还是能够完全把两部分的点分开，如下图：

![](https://pic1.zhimg.com/v2-5e68140c9f3b2fec91649929f90e6138_b.png)

对于这个问题，我们是否应该让D变得弱一点，减弱它的分类能力，但是从理论上讲，为了让它能够有效的区分真假图片，我们又希望它能够powerful，所以这里就产生了矛盾。

还有可能的原因是，虽然两个分布都是高维的，但是两个分布都十分的窄，可能交集相当小，这样也会导致JS divergence算出来=log2，约等于没有交集。

解决的一些方法，有添加噪声，让两个分布变得更宽，可能可以增大它们的交集，这样JS divergence就可以计算，但是随着时间变化，噪声需要逐渐变小。

还有一个问题叫Mode Collapse，如下图：

![](https://pic4.zhimg.com/v2-ab84e45babe1a71bc19963b0455bfdcf_b.png)

这个图的意思是，data的分布是一个双峰的，但是学习到的生成分布却只有单峰，我们可以看到模型学到的数据，但是却不知道它没有学到的分布。

造成这个情况的原因是，KL divergence里的两个分布写反了

![](https://pic2.zhimg.com/v2-47e26e6096ef6967f064986c23e373e5_b.png)

这个图很清楚的显示了，如果是第一个KL divergence的写法，为了防止出现无穷大，所以有![](http://www.zhihu.com/equation?tex=P_%7Bdata%7D "P\_{data}")出现的地方都必须要有![](http://www.zhihu.com/equation?tex=P_G "P\_G")覆盖，就不会出现Mode Collapse

