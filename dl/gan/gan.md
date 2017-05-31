本期我们来聊聊GANs（Generativeadversarial networks，对抗式生成网络，也有人译为生成式对抗网络）。GAN最早由Ian Goodfellow于2014年提出，以其优越的性能，在不到两年时间里，迅速成为一大研究热点。

  


**GANs与博弈论**

  


GANs是一类生成模型，从字面意思不难猜到它会涉及两个“对手”，一个称为Generator（生成者），一个称为Discriminator（判别者）。Goodfellow最近arxiv上挂出的GAN tutorial文章中将它们分别比喻为伪造者（Generator）和警察（Discriminator）。伪造者总想着制造出能够以假乱真的钞票，而警察则试图用更先进的技术甄别真假钞票。两者在博弈过程中不断升级自己的技术。  


![](https://mmbiz.qpic.cn/mmbiz_jpg/yAnhaHNJib1ZKfqLlhy3UB99BHy3n6IoRdW0eV1NtFDCQO0ZAm6LtibKMLE0LazxnEn1j6yo83jgo9F0DWJUuFIQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)  


从博弈论的角度来看，如果是零和博弈（zero-sum game），两者最终会达到纳什均衡（Nash equilibrium），即存在一组策略\(g, d\)，如果Generator不选择策略g，那么对于Discriminator来说，总存在一种策略使得Generator输得更惨；同样地，将Generator换成Discriminator也成立。囚徒的困境就是一个典型的零和博弈。

  




如果GANs定义的lossfunction满足零和博弈，并且有足够多的样本，双方都有充足的学习能力情况，在这种情况下，Generator和Discriminator的最优策略即为纳什均衡点，也即：Generator产生的都是“真钞”（材料、工艺技术与真钞一样，只是没有得到授权...），Discriminator会把任何一张钞票以1/2的概率判定为真钞。

![](https://mmbiz.qpic.cn/mmbiz_png/yAnhaHNJib1ZKfqLlhy3UB99BHy3n6IoRogRSL9r040v4Aw85nDP6TSyoF2gTyFd1qBcQciaRpkVZ4eEn1K9Xk5A/0?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)  


讲到这里，今天的推送基本就讲完了...

  


额，至少思想和框架都介绍完了，具体怎么操作呢，大家可以先想一想，然后再往下看。

  


下面的内容，我们分为三个部分：

1. 为什么要研究生成模型（Generative model）？只是出于好玩吗？

2. GANs——IanGoodfellow在2014年提出来的版本，以及它的一些演化（GANs有很多种，这里主要介绍最初的版本，其他版本将在以后介绍）

3. GANs有什么优缺点？

  


以下内容根据Goodfellow 2014年发表的GAN论文和代码（arxiv 1406.2661）以及近期arxiv上挂出的NIPS tutorial（arxiv 1701.00160 ）整理。

**  
**

**为什么要研究生成模型**

  


先来讲讲为什么要研究生成模型，首先，生成模型真的很好玩......

还是来看看Ian Goodfellow怎么说吧。

* 高维概率分布在实际应用中非常常见。训练生成模型，并用生成模型采样可以检验我们对高维概率分布的表示和操纵能力。

* 生成模型能够嵌入到增强学习（reinforcement learning）的框架中。例如用增强学习求解规划问题，可以用生成模型学习一个条件概率分布，agent可以根据生成模型对不同actions的响应，选择尽可能好的action。

* 生成模型一般是非监督或者半监督模型，能够处理数据缺失问题，并且还可以用于预测缺失的数据。

* 生成模型，特别是GANs，能处理多模态输出（multi-modal）的问题。多模态输出问题（对于离散输出来说，就是多类标问题【multi-label】）是很多机器学习算法没办法直接处理的问题，很多机器学习算法的loss定义为均方误差，它实际上将多种可能的输出做了平均。下图给出了一个预测视频的下一帧的例子，使用MSE训练的结果，图像模糊了（它对所有可能的结果做了平均），而使用GANs训练的结果，则不存在这种问题。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  


* 某些任务在本质上要求根据某些分布产生样本。例如：用低分辨率图像产生高分辨率图像；用草图生成真实图像；根据卫星图像产生地图等。如下图所示。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  


  


**最初版本的GANs模型**

  


GANs在很多应用问题上取得非常好的效果，让生成模型进入人们的视野。前面已经介绍了GANs的思想，接下来，我们就来进一步介绍GANs是怎么做的。  


  


首先，我们需要定义博弈的双方：Generator和Discriminator。

  


以训练生成图像为例，Generator的任务就是根据输入产生图像。而Discriminator则是判断图像是真实图像还是Generator生成的图像。

  


Generator和Discriminator具体怎么做可以自己定义。

  


这里根据我的理解，举个例子：我们知道卷积神经网络能够实现图像抽象特征的提取，如果我们定义Generator的输入为“图像的特征”（训练时没有“图像的特征”，输入可以是随机的噪声向量），那么可以用含有反卷积层或者上采样层的网络实现图像的重构，也就是图像的生成。

  


Discriminator是一个二元分类器，输入是图像，输出是两类：“自然”图像/Generator产生的图像。这里说的“自然”图像并不一定是自然图像，可以是合成的图像，人眼看上去图像是自然的。二元分类器有很多种，卷积神经网络是一个不错的选择。

  


其次，要定义loss function才能训练。前面说了，GANs可以看成一个博弈，那么博弈双方都会有cost（代价），如果是零和博弈，那么双方的cost之和为0。Discriminator是一个分类器，它的loss可以定义用交叉熵来定义：

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

如果是零和博弈，那么Generator的loss就定义为：

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  


整个优化问题就是一个minmax博弈：

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  


Goodfellow在2014年的论文中证明了在理想情况下（博弈双方的学习能够足够强、拥有足够的数据、每次迭代过程中，固定Generator，Discriminator总能达到最优），这样构造的GANs会收敛到纳什均衡解。基于此，在2014年的论文里面，作者提出的算法是，每次迭代过程包括两个步骤：更新k次Discriminator（k&gt;=1）；更新1次Generator。也就是，应该让Discriminator学得更充分一些。PS：2016 NIPS tutorial中，作者指出，每次迭代对博弈双方同时进行（随机）梯度下降，在实际操作中效果最好。

  


然而，这样定义的零和博弈在实际中效果并不好，实际训练的时候，Discriminator仍然采用前面介绍的loss，但Generator采用下面的loss：

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  


也就是说，Generator的loss只考虑它自身的交叉熵，而不用全局的交叉熵。这样做能够确保任一博弈失利的一方仍有较强的梯度以扭转不利局面。作者称采用这个loss的博弈为Non-saturating heuristic game。采用这个版本的loss，整个博弈不再是零和博弈，没有理论保证它会收敛达到纳什均衡。

  




**最大似然博弈GANs**

  


原始版本的GANs模型并不是通过最大化似然函数（也即最小化KL散度）来训练的。Goodfellow在14年的另一篇论文（https://arxiv.org/abs/1412.6515）中证明了，若每次迭代Discriminator都达到最优，Generator采用下面的loss，则GANs等价于最大化似然函数：  


![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  


至此，本文总共出现了三种loss，它们的函数图像长这样：

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  


**  
**

**GANs实验效果**

  


根据Goodfellow 2014年的论文，GANs在MNIST和TFD数据集上训练，产生的样本更接近于真实分布（参看下图中的Table 1）。然而，虽然效果更好，GANs也存在一些问题（如下面的Figure 29和30），话不多说，看图。  


![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  
![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  
![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)  
![](https://mmbiz.qpic.cn/mmbiz_jpg/yAnhaHNJib1ZKfqLlhy3UB99BHy3n6IoRPF9iahG22bAc40fiaSf6O6IibOxtbkbnmh1tNlVb1azrEOvXeD2ERSBtg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)  
![](https://mmbiz.qpic.cn/mmbiz_jpg/yAnhaHNJib1ZKfqLlhy3UB99BHy3n6IoRYTkM4ULj2jvkQadJI6xL2qNpPibOTlVicuGNcDOtFepKJLHcddkqm6Gw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)  


**  
**

**GANs优缺点**

  


优点：  


* 相比于FVBNs（全可见信念网络，Fully visible belief networks）模型，GANs能并行生成样本，不需要逐维产生。

* 相比于玻尔兹曼机（Boltzmann machines）、非线性ICA（non-linear ICA）等生成模型，GANs对Generator的设计的限制很少。

* 理论上保证了某些GANs能够收敛到纳什均衡。

* 最重要的是，相比于其他生成模型，GANs产生的样本更好。（这一条是主观判断...）



缺点：

* 训练GANs实际上是在找纳什均衡解，这比优化一个目标函数要困难。

* GANs生成的图像比较sharp，也就是说，它倾向于生成相似的图像。作者在2016 NIPS tutorial中指出，这个缺陷与采用何种KL散度作为loss无关，而可能是与训练过程有关。详细的讨论可以参看参考文献2的3.2.5和5.1.1章节。PS：这是一个非常重要的问题，如果想深入理解GANs，请务必阅读原文详细了解。



PS： 如果看完你还是没理解是怎么做的，那就看代码吧，网上的代码很多，点击阅读原文（Ian Goodfellow 2014年论文的代码）或者看文末给的代码链接（有simple版本的代码）。阅读源码对于理解文献的方法有很大的好处（前提是源码好理解...）；看完论文自己复现论文结果对于进一步理解论文，甚至改进方法都有很大的帮助。这也是我目前在做的事情。

  


**参考文献**

  


1. Goodfellow I,Pouget-Abadie J, Mirza M, et al. Generative adversarial nets\[C\]//Advances inNeural Information Processing Systems. 2014: 2672-2680.

2. Goodfellow I.NIPS 2016 Tutorial: Generative Adversarial Networks\[J\]. arXiv preprintarXiv:1701.00160, 2016.

3. Goodfellow IJ. On distinguishability criteria for estimating generative models\[J\]. arXivpreprint arXiv:1412.6515, 2014.



**GANs代码**

  


1. Theano+pylearn2版本（Goodfellow提供）： https://github.com/goodfeli/adversarial

2. Keras版本： https://github.com/jhayes14/GAN

3. Tensorflow版本（含GANs+VAEs+DRAW）： https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW





首先来看看 GAN 现在能做到哪些惊艳的事呢？  




![](http://mmbiz.qpic.cn/mmbiz_png/qrpuZA2scNr86w3hAZ8S8w3n5eHVMZpXrSJRjUuibzCbnib1jWRlNj23iaHico379mlV3fiaMYK7OdS0VWpThDxBKLQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

GAN 可以被用来学习生成各种各样的字体——也就是说，以后字迹辨认这种侦查手段很可能就不靠谱啦！这个工作还有很多动图，在 GitHub 上搜 zi2zi 这个 project 就可以。

![](http://mmbiz.qpic.cn/mmbiz_png/qrpuZA2scNr86w3hAZ8S8w3n5eHVMZpXvEIYhMyUozTicQ2mOP4ib2SulRj2gV7szm3NicuN8OUnfnVB44sI70GLg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

有了 GAN，以后就不怕灵魂画手了！左边这么简单的“简笔画”，居然也可以直接生成出对应的蕴含丰富含义的完整图画呢。这个工作来自\[24\]，同时还可以用来做一些修正，比如可以把春天的山变成被白雪覆盖尖端的山——而这一切只需要一点点白色的涂抹（可以参考这个工作\[24\]的官方页面）。



![](http://mmbiz.qpic.cn/mmbiz_png/qrpuZA2scNr86w3hAZ8S8w3n5eHVMZpXZSahz0Yr8RFeo9lPkhJws2VZc2WlGgEJ4w1KoSjdDgas55CkwXiawNA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

有了 GAN，不仅仅可以在有辅助的情况下作画，还可以在无辅助的情况下完成很多修饰！比如从分割图变成真实照片，从黑白图变成彩色图，从线条画变成富含纹理、阴影和光泽的图……这些都是用 pix2pix\[21\] 里的方法生成的。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

去年非常火爆的应用，脸萌当然也难不倒 GAN！想拥有自己的专属Q版定制头像吗？快去看看\[22\] 吧（[这篇工作以前也已经写过哦，传送门点此](http://mp.weixin.qq.com/s?__biz=MzAwMjM3MTc5OA==&mid=2652692475&idx=1&sn=4bddcff7723890888b988e8eeb8e5fef&chksm=81230642b6548f54f301602c144ba5eb1884f9cb9c18a64b9cee2d56bf576d1bc8e2d62e33ac&scene=21#wechat_redirect)）。



![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

当然还少不了前段时间刷爆社交网络的“梵高在春天醒来”、“马变斑马”“四季变换”的工作啦。来自依然也写过的 CycleGAN\[9\]，[传送门点此](http://mp.weixin.qq.com/s?__biz=MzAwMjM3MTc5OA==&mid=2652692559&idx=1&sn=992c5b9f4d12d16fa3df78cae954172e&chksm=812339f6b654b0e0a69d119caa2573a1ef4dca2e2493312a25d3e201d4b05e341b0a1d9c475b&scene=21#wechat_redirect)。

  


这些惊艳的工作基本都是2016年8月甚至10月以后的，也就是 GAN 被提出两年后。这是因为，虽然 GAN 有非常吸引人的性质，想要训练好它并不容易。经过两年的摸索、思考与尝试，才有了如今的积累和突破。

那么这个非常吸引人的 GAN 是什么样呢。其实 GAN 最初让人“哇”的地方在于，作为一个生成模型，GAN 就像魔术师变魔术一样，只需要一个噪音（噪音向量），就可以生成一只兔子！

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

而想要成为一个成功的欺骗观众的魔术师并不容易，GAN 也是在不断地失败、穿帮、磨练技艺中成长起来的！要知道，观众们见过很多兔子，如果变出来的东西根本不像兔子，或者变不出来，这个魔术就很失败，观众就不会买账。在这样反复的练习中，作为魔术师的 GAN 扮演的是生成模型的角色，目的是要不断地提高自己的魔术水平，从而变出更活灵活现的兔子；而观众扮演的是一种判别模型的角色，目的是考察和激励魔术师提高自己的水平。但是这种激励是通过批评或者惩罚的方式完成的。

**严格来说，**一个 GAN 框架，最少（但不限于）拥有两个组成部分，一个是生成模型 G，一个是判别模型 D。在训练过程中，会把生成模型生成的样本和真实样本随机地传送一张（或者一个 batch）给判别模型 D。判别模型 D 的目标是尽可能正确地识别出真实样本（输出为“真”，或者1），和尽可能正确地揪出生成的样本，也就是假样本（输出为“假”，或者0）。**这两个目标分别对应了下方的目标函数的第一和第二项。**而生成模型的目标则和判别模型相反，就是尽可能最小化判别模型揪出它的概率。这样 G 和 D 就组成了一个 min-max game，在训练过程中双方都不断优化自己，直到达到平衡——双方都无法变得更好，也就是假样本与真样本完全不可区分。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

  


**通过这样的巧妙设计，GAN 就拥有了一个非常吸引人的性质。**GAN 中的 G 作为生成模型，不需要像传统图模型一样，需要一个严格的生成数据的表达式。这就避免了当数据非常复杂的时候，复杂度过度增长导致的不可计算。同时，它也不需要 inference 模型中的一些庞大计算量的求和计算。它唯一的需要的就是，一个噪音输入，一堆无标准的真实数据，两个可以逼近函数的网络。

但是天下没有免费的午餐，这样简单的要求使得 GAN 的自由度非常大。换句话说，GAN 的训练就会很容易失去方向，变得野蛮生长。于是，早期的 GAN 经常出现如下让人崩溃的现象：

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

这些现象其实是 GAN 中存在的三个难点或者说问题交织导致的。个人觉得，**首当其中的难点一就是，深度神经网络自身的难训练和不稳定。**虽然原始 GAN 的理论中，并不要求 G 和 D 都是神经网络，只需要是能拟合相应生成和判别的函数就可以。但是这恰好是深度神经网络擅长的事情嘛，所以大家就都用神经网络作为 G 和 D 了。但是神经网络的选择、设计和训练还充满了艺术的惊喜与生活的不确定性，这也直接为 GAN 的训练带来了困难。加之本来 GAN 就缺乏指导，所以就有了一系列可以被归结为解决这一方向问题的工作。我将这类工作概括为 Partial Guidance, Fine-Grained Guidance 和 Special Architecture。  


先来看，Partial Guidance。Partial Guidance 中我概括的几个重要工作，都是为原始 GAN 加上一些显式的外部信息，比如用户的额外输入，比如类别信息等等。包含的工作有：  




![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

**Conditional GAN\[15\]，也叫 CGAN，几乎是原始 GAN\[2\] 后的第一份工作**，想法非常简单，既然你的信息不够，我就把你原始的生成过程变成基于某些额外信息的生成。这样就相当于给你提供了一些 hint，所以公式如下：  




![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

可以看到，D 和 G 去拟合的分布都变成了条件概率分布。在 CGAN 的工作中，这个额外的 y 信息，是通过在输入层直接拼接样本与 y 信息的向量而实现的。具体使用的 y 信息有 one-hot vector，也有图像（也就是基于另一个图像去生成）。这个 y 信息的选择其实十分灵活，在后期的工作中也依然很常见，毕竟是一种非常直观有效的加入 label 信息的方式。

**第二个这方面的工作是由 OpenAI 提出的 Improved GAN**\[19\]，其中重点提出了两个训练 GAN 的技巧，feature matching 和 minibatch discrimination。feature matching 是指，既然 G 和 D 的训练不够稳定，常常 D 太强，G 太弱，那么不如就把 D 网络学到的特征直接“传”给 G，让 G 不仅能知道 D 的输出，还能知道 D 是基于什么输出的。所以就有了如下的新的目标函数：



![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

也就是说，现在的 D 直接就是神经网络的某一个中间层了。这个方法在实验中发现对于训练的稳定度提升非常有帮助。与此同时，他们还提出了第二个方法，叫 minibatch discrimination：



![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

这其实是让 D 在判断当前传给它的样本是真是假的同时，不要只关注当前的，也要关注其他的样本。这会增加 D 判断样本时候的多样性，从而增加了 G 生成样本的多样性，因为它不再会只基于一种“逼真”样本的方向来改进自己。

**第三个工作是来自 UC Berkeley 的 iGAN/GVM**\[24\]，也是开篇介绍 GAN 应用中的解放灵魂画手的工作。他们的工作中蕴含了两种指导信息，一个是用户的输入，比如蓝色的笔触，比如绿色的线条，比如图像改变的方向和程度（拉伸、变形）。但是如果只利用这样的信息，生成的图像往往比较模糊，比如鞋子的纹理变得不够清晰。为此它们的解决办法是为在生成的鞋子的形状上“贴”上原始真实图片中的高清纹理。所以难点就是如何让“贴”的过程变得可靠，不能“贴”出区域，也不能“贴”少了。他们在此利用了差值空间中的光场信息，从而能捕捉到相邻差值空间中的点对点映射关系，也就可以基于这样的映射，迭代“贴”上纹理，直到最后一步：  




![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

**iGAN 的工作之后，他们又继续做了 pix2pix 的工作**\[21\]，用于生成一些图像两两之间的“变换”。也就是开篇介绍的，“从分割图变成真实照片，从黑白图变成彩色图，从线条画变成富含纹理、阴影和光泽的图”，还有第一个 zi2zi 的字体变换，也是基于这个 pix2pix 的工作\[21\]。pix2pix 里，将 D 的输出从一张图片变成了一对图片，所以 D 的任务就变成了去判断当前的两张图片是否是一个“真实”的“变换”。比如我们的需求是给一个黑白的 Hello Kitty 上色，那么 pix2pix 的框架大概如下：



![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

**而 Partial Guidance 中的最后一个重要任务就是非常新的 GP-GAN\[25\]**，目标是将直接复制粘贴过来的图片，更好地融合进原始图片中，做一个 blending 的事情。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

这个过程非常像 iGAN，也用到了类似 iGAN 中的一些约束，比如 color constraint。另一方面，这个工作也有点像 pix2pix，因为它是一种有监督训练模型，在 blending 的学习过程中，会有一个有监督目标和有监督的损失函数。



![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

除了 Partial Guidance 这种非常显式的“半监督”（不是严格意义上的半监督）信息，过去也有很多工作**让 GAN 的生成过程拆解到多步，从而实现“无监督”的 Fine-grained Guidance**。个人总结了以下一些重要工作：

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

第一篇 LAPGAN 是来自 Facebook\[16\]，是第一篇将层次化或者迭代生成的思想运用到 GAN 中的工作。在原始 GAN\[2\] 和后来的 CGAN\[15\] 中，GAN 还只能生成 16\*16, 28\*28, 32\*32 这种低像素小尺寸的图片。而这篇工作\[16\] 是首次成功实现 64\*64 的图像生成。思想就是，与其一下子生成这么大的（包含信息量这么多），不如一步步由小转大，这样每一步生成的时候，可以基于上一步的结果，而且还只需要“填充”和“补全”新大小所需要的那些信息。这样信息量就会少很多：



![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

而为了进一步减少信息量，他们甚至让 G 每次只生成“残差”图片，生成后的插值图片与上一步放大后的图片做加法，就得到了这一步生成的图片。

**第二篇 Fine-grained Guidance 方面的工作\[18\]**讨论的是从 Text 生成 Image，比如从图片标题生成一个具体的图片。这个过程需要不仅要考虑生成的图片是否真实，还应该考虑生成的图片是否符合标题里的描述。比如要标题形容了一个黄色的鸟，那么就算生成的蓝色鸟再真实，也是不符合任务需求的。为了捕捉或者约束这种条件，他们提出了 matching-aware discriminator 的思想，让本来的 D 的目标函数中的两项，扩大到了三项：  


![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

**第三篇这方面的工作\[20\]**可以粗略认为是 LAPGAN\[16\] 和 matching-aware\[18\] 的结合。**他们提出的 StackGAN**\[20\] 做的事情从标题生成鸟类，但是生成的过程则是像 LAPGAN 一样层次化的，从而实现了 256\*256 分辨率的图片生成过程。StackGAN 将图片生成分成两个阶段，阶段一去捕捉大体的轮廓和色调，阶段二加入一些细节上的限制从而实现精修。这个过程效果很好，甚至在某些数据集上以及可以做到以假乱真：  


![](http://mmbiz.qpic.cn/mmbiz_png/qrpuZA2scNrFiawsJqjECSCiad3fdw13oemR0lckIwGFUuHJLw9GumTwqjwG9BnnOLRI6IYd1rJ6gmp6v94NECCw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

**最后一个这方面的工作\[26\]**，当时也因为效果逼真和工作夯实，引发了社交网络上和学术圈的广泛关注，那就是**去年年底的 PPGN\[26\]，现在已被 CVPR 2017 接收**。PPGN 也主张不要一次生成一张完整的图片，而是要用一个迭代过程不断地调整和完善。与 LAPGAN 和 StackGAN 不同的是，PPGN 使用了 Denoising AutoEncoder（DAE）的过程实现迭代，并在其网络结构中也多次体现了迭代和层次化的思想。

![](http://mmbiz.qpic.cn/mmbiz_png/qrpuZA2scNrFiawsJqjECSCiad3fdw13oeicn48mle7yVkz8OvkQDxGWgoxEgOAgkMickPm9VGe98LcHL9N4e9rEPA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

  


![](http://mmbiz.qpic.cn/mmbiz_gif/qrpuZA2scNrFiawsJqjECSCiad3fdw13oedeeow4eUfAkCqwQx9DRc9q7Ga6UtibOG2DE6OGsvNw6g7QOEzThg1Bw/0?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

**针对 GAN 的第一大难点，也就是神经网络本身训练的不稳定和难调参，也有许多工作提出了一些特殊结构，来改善这些情况。**

![](http://mmbiz.qpic.cn/mmbiz_png/qrpuZA2scNrFiawsJqjECSCiad3fdw13oeB7ZVeGWaMOicM8g2ia0Xz6YxibfibnqhG6TtNUolViaWYRDsSmkrMJT7JLg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

将 GAN 成功从 MNIST 的多层感知机（MLP）结构扩展到卷积神经网络结构的**就是 DCGAN 这篇工作**\[17\]。这篇工作中，他们提出了一组卷积神经网络，不仅使得可以 GAN 可以在 celebA 和 LSUN 这种现实世界的真实大规模数据集上训练，还使得 batchnorm 等 trick 也被成功运用。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

虽然 DCGAN 的一度成为了 GAN 工作的标准，统治了大半年的江湖。**但是随后出现的效果非常惊艳的 pix2pix**\[21\] 中却指出了 U-Net 结构的优势。pix2pix 中 G 和 D 使用的网络都是 U-Net 结构，是一种 encoder-decoder 完全对称的结构，并且在这样的结构中加入了 skip-connection 的使用。

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

这个结构对于生成效果至关重要，其也被后续的一些工作采用\[9\]\[11\] 。skip-connection 不仅使得梯度传导更通畅，网络训练更容易，也因为这类工作多数是要学习图片之间的映射，那么让 encoder 和 decoder 之间一一对应的层学到尽可能匹配的特征将会对生成图片的效果产生非常正面的影响。类似的讨论可以见 \[11\]。

**最后要指出的也是刚才就提到的 GP-GAN**\[25\] 的工作。在这个工作中，它们提出了 blending GAN 的模块，虽然也是基于 encoder-decoder 的结构，但是略有不同的地方是，在两者中间加入了一个 fully-connected layer：

![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

这个全连接层的特殊之处在于，并不是卷积神经网络中的 channel-wise FCN，而是彻底全连接。这样的好处是可以传递更多的全局信息，使得有监督学习变得更加有效。

  


![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

以上就是本次分享中的上半部分，GAN 的基础和 GAN 的难点一、以及对应的解决方案。关于 GAN 的难点二、三，和对应解决方案，会在文字版（下）中整理完毕。

最后再次附上这份文字版干货对应的完整 slides 和回顾视频。**只要关注“程序媛的日常”微信公众号，回复“原理篇”就可以获取126页完整 PDF 和分享视频啦！**

  


\[1\] Arjovsky and Bottou, “Towards Principled Methods for Training Generative Adversarial Networks”. ICLR 2017. 

\[2\] Goodfellow et al., “Generative Adversarial Networks”. ICLR 2014. 

\[3\] Che et al., “Mode Regularized Generative Adversarial Networks”. ICLR 2017. 

\[4\] Zhao et al., “Energy-based Generative Adversarial Networks”. ICLR 2017. 

\[5\] Berthelot et al., “BEGAN: Boundary Equilibrium Generative Adversarial Networks”. arXiv preprint 2017. 

\[6\] Sønderby, et al., “Amortised MAP Inference for Image Super-Resolution”. ICLR 2017. 

\[7\] Arjovsky et al., “Wasserstein GANs”. ICML 2017. 

\[8\] Villani, Cedric. “Optimal transport: old and new”, volume 338. Springer Science & Business Media, 2008.

\[9\] Jun-Yan Zhu\*, Taesung Park\*, Phillip Isola, Alexei A. Efros. “Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”. arXiv preprint 2017. 

\[10\] Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, Jiwon Kim. “Learning to Discover Cross-Domain Relations with Generative Adversarial Networks”. ICML 2017. 

\[11\] Zili Yi, Hao Zhang, Ping Tan, Minglun Gong. “DualGAN: Unsupervised Dual Learning for Image-to-Image Translation”. arXiv preprint 2017. 

\[12\] Jeff Donahue, Philipp Krähenbühl, Trevor Darrell. “Adversarial Feature Learning”. ICLR 2017. 

\[13\] Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Olivier Mastropietro, Alex Lamb, Martin Arjovsky, Aaron Courville. “Adversarially Learned Inference”. ICLR 2017. 

\[14\] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville. “Improved Training of Wasserstein GANs”. arXiv preprint 2017.

\[15\] Mehdi Mirza, Simon Osindero. “Conditional Generative Adversarial Nets”. arXiv preprint 2014. 

\[16\] Emily Denton, Soumith Chintala, Arthur Szlam, Rob Fergus. “Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks”. arXiv preprint 2015. 

\[17\] Alec Radford, Luke Metz, Soumith Chintala. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”. ICLR 2016.

\[18\] Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, Honglak Lee. “Generative Adversarial Text to Image Synthesis”. ICML 2016. 

\[19\] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen. “Improved Techniques for Training GANs”. arXiv preprint 2016. 

\[20\] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaolei Huang, Xiaogang Wang, Dimitris Metaxas. “StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks”. arXiv preprint 2016. 

\[21\] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros. “Image-to-Image Translation with Conditional Adversarial Networks”. CVPR 2017.

\[22\] Yaniv Taigman, Adam Polyak, Lior Wolf. “Unsupervised Cross-Domain Image Generation”. ICLR 2017. 

\[23\] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. “Generative Adversarial Nets”. NIPS 2014. 

\[24\] Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman and Alexei A. Efros. “Generative Visual Manipulation on the Natural Image Manifold”, ECCV 2016. 

\[25\] Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang. “GP-GAN: Towards Realistic High-Resolution Image Blending”. arXiv preprint 2017. 

\[26\] Anh Nguyen, Jeff Clune, Yoshua Bengio, Alexey Dosovitskiy, Jason Yosinski. “Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space”. CVPR 2017.

