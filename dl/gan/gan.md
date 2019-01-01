本期我们来聊聊GANs（Generativeadversarial networks，对抗式生成网络，也有人译为生成式对抗网络）。GAN最早由Ian Goodfellow于2014年提出，以其优越的性能，在不到两年时间里，迅速成为一大研究热点。

**GANs与博弈论**

GANs是一类生成模型，从字面意思不难猜到它会涉及两个“对手”，一个称为Generator（生成者），一个称为Discriminator（判别者）。Goodfellow最近arxiv上挂出的GAN tutorial文章中将它们分别比喻为伪造者（Generator）和警察（Discriminator）。伪造者总想着制造出能够以假乱真的钞票，而警察则试图用更先进的技术甄别真假钞票。两者在博弈过程中不断升级自己的技术。

从博弈论的角度来看，如果是零和博弈（zero-sum game），两者最终会达到纳什均衡（Nash equilibrium），即存在一组策略\(g, d\)，如果Generator不选择策略g，那么对于Discriminator来说，总存在一种策略使得Generator输得更惨；同样地，将Generator换成Discriminator也成立。囚徒的困境就是一个典型的零和博弈。

如果GANs定义的lossfunction满足零和博弈，并且有足够多的样本，双方都有充足的学习能力情况，在这种情况下，Generator和Discriminator的最优策略即为纳什均衡点，也即：Generator产生的都是“真钞”（材料、工艺技术与真钞一样，只是没有得到授权...），Discriminator会把任何一张钞票以1/2的概率判定为真钞。

讲到这里，今天的推送基本就讲完了...

额，至少思想和框架都介绍完了，具体怎么操作呢，大家可以先想一想，然后再往下看。

下面的内容，我们分为三个部分：

1. 为什么要研究生成模型（Generative model）？只是出于好玩吗？

2. GANs——IanGoodfellow在2014年提出来的版本，以及它的一些演化（GANs有很多种，这里主要介绍最初的版本，其他版本将在以后介绍）

3. GANs有什么优缺点？

以下内容根据Goodfellow 2014年发表的GAN论文和代码（arxiv 1406.2661）以及近期arxiv上挂出的NIPS tutorial（arxiv 1701.00160 ）整理。

---

**为什么要研究生成模型**

先来讲讲为什么要研究生成模型，首先，生成模型真的很好玩......

还是来看看Ian Goodfellow怎么说吧。

* 高维概率分布在实际应用中非常常见。训练生成模型，并用生成模型采样可以检验我们对高维概率分布的表示和操纵能力。

* 生成模型能够嵌入到增强学习（reinforcement learning）的框架中。例如用增强学习求解规划问题，可以用生成模型学习一个条件概率分布，agent可以根据生成模型对不同actions的响应，选择尽可能好的action。

* 生成模型一般是非监督或者半监督模型，能够处理数据缺失问题，并且还可以用于预测缺失的数据。

* 生成模型，特别是GANs，能处理多模态输出（multi-modal）的问题。多模态输出问题（对于离散输出来说，就是多类标问题【multi-label】）是很多机器学习算法没办法直接处理的问题，很多机器学习算法的loss定义为均方误差，它实际上将多种可能的输出做了平均。下图给出了一个预测视频的下一帧的例子，使用MSE训练的结果，图像模糊了（它对所有可能的结果做了平均），而使用GANs训练的结果，则不存在这种问题。

![](/images/dl/gan/import.png)

* 某些任务在本质上要求根据某些分布产生样本。例如：用低分辨率图像产生高分辨率图像；用草图生成真实图像；根据卫星图像产生地图等。如下图所示。

![](/images/dl/gan/import2.png)

**最初版本的GANs模型**

GANs在很多应用问题上取得非常好的效果，让生成模型进入人们的视野。前面已经介绍了GANs的思想，接下来，我们就来进一步介绍GANs是怎么做的。

首先，我们需要定义博弈的双方：Generator和Discriminator。

以训练生成图像为例，Generator的任务就是根据输入产生图像。而Discriminator则是判断图像是真实图像还是Generator生成的图像。

Generator和Discriminator具体怎么做可以自己定义。

这里根据我的理解，举个例子：我们知道卷积神经网络能够实现图像抽象特征的提取，如果我们定义Generator的输入为“图像的特征”（训练时没有“图像的特征”，输入可以是随机的噪声向量），那么可以用含有反卷积层或者上采样层的网络实现图像的重构，也就是图像的生成。

Discriminator是一个二元分类器，输入是图像，输出是两类：“自然”图像/Generator产生的图像。这里说的“自然”图像并不一定是自然图像，可以是合成的图像，人眼看上去图像是自然的。二元分类器有很多种，卷积神经网络是一个不错的选择。

其次，要定义loss function才能训练。前面说了，GANs可以看成一个博弈，那么博弈双方都会有cost（代价），如果是零和博弈，那么双方的cost之和为0。Discriminator是一个分类器，它的loss可以定义用交叉熵来定义：
$$J^{(D)}(\theta^{(D)},\theta^{(G)})=-\frac {1}{2}E_{xP\sim _{data}}logD(x)-\frac {1}{2}E_zlog(1-D(G(z)))$$
如果是零和博弈，那么Generator的loss就定义为：

$$J^{(D)}(\theta^{(D)},\theta^{(G)})=-J^{eta^{(D)},\theta^{(G)})=\frac {1}{2}E_{xP\sim _{data}}logD(x)+\frac {1}{2}E_zlog(1-D(G(z)))$$
整个优化问题就是一个minmax博弈
![](/images/dl/gan/minmax.png)

Goodfellow在2014年的论文中证明了在理想情况下（博弈双方的学习能够足够强、拥有足够的数据、每次迭代过程中，固定Generator，Discriminator总能达到最优），这样构造的GANs会收敛到纳什均衡解。基于此，在2014年的论文里面，作者提出的算法是，每次迭代过程包括两个步骤：更新k次Discriminator（k&gt;=1）；更新1次Generator。也就是，应该让Discriminator学得更充分一些。PS：2016 NIPS tutorial中，作者指出，每次迭代对博弈双方同时进行（随机）梯度下降，在实际操作中效果最好。

然而，这样定义的零和博弈在实际中效果并不好，实际训练的时候，Discriminator仍然采用前面介绍的loss，但Generator采用下面的loss：

$$J^{(D)}(\theta^{(D)},\theta^{(G)})=-\frac {1}{2}E_zlog(1-D(G(z)))$$

也就是说，Generator的loss只考虑它自身的交叉熵，而不用全局的交叉熵。这样做能够确保任一博弈失利的一方仍有较强的梯度以扭转不利局面。作者称采用这个loss的博弈为Non-saturating heuristic game。采用这个版本的loss，整个博弈不再是零和博弈，没有理论保证它会收敛达到纳什均衡。

**最大似然博弈GANs**

原始版本的GANs模型并不是通过最大化似然函数（也即最小化KL散度）来训练的。Goodfellow在14年的另一篇论文（[https://arxiv.org/abs/1412.6515）中证明了，若每次迭代Discriminator都达到最优，Generator采用下面的loss，则GANs等价于最大化似然函数：](https://arxiv.org/abs/1412.6515）中证明了，若每次迭代Discriminator都达到最优，Generator采用下面的loss，则GANs等价于最大化似然函数：)

$$J^{(G)}=-\frac{1}{2}E_zexp(\sigma ^{-1}(D(G(z))))$$

至此，本文总共出现了三种loss，它们的函数图像长这样：

![](/images/dl/gan/gan3loss.png)**      
**

**GANs实验效果**

根据Goodfellow 2014年的论文，GANs在MNIST和TFD数据集上训练，产生的样本更接近于真实分布（参看下图中的Table 1）。然而，虽然效果更好，GANs也存在一些问题（如下面的Figure 29和30），话不多说，看图。


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

1. Theano+pylearn2版本（Goodfellow提供）： [https://github.com/goodfeli/adversarial](https://github.com/goodfeli/adversarial)

2. Keras版本： [https://github.com/jhayes14/GAN](https://github.com/jhayes14/GAN)

3. Tensorflow版本（含GANs+VAEs+DRAW）： [https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW](https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW)

**各种GAN**
# The GAN Zoo

<p align="center"><img width="40%" src="The_GAN_Zoo.jpg" /></p>

Every week, new GAN papers are coming out and it's hard to keep track of them all, not to mention the incredibly creative ways in which researchers are naming these GANs! So, here's a list of what started as a fun activity compiling all named GANs!

<p align="center"><img width="50%" src="cumulative_gans.jpg" /></p>

You can also check out the same data in a tabular format with functionality to filter by year or do a quick search by title [here](https://github.com/hindupuravinash/the-gan-zoo/blob/master/gans.tsv).

Contributions are welcome. Add links through pull requests or create an issue to lemme know something I missed or to start a discussion.

Check out [Deep Hunt](https://deephunt.in) - my weekly AI newsletter for this repo as [blogpost](https://medium.com/deep-hunt/the-gan-zoo-79597dc8c347) and follow me on [Twitter](https://www.twitter.com/hindupuravinash).

* 3D-GAN - [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://arxiv.org/abs/1610.07584) ([github](https://github.com/zck119/3dgan-release))
* 3D-IWGAN - [Improved Adversarial Systems for 3D Object Generation and Reconstruction](https://arxiv.org/abs/1707.09557) ([github](https://github.com/EdwardSmith1884/3D-IWGAN))
* 3D-RecGAN - [3D Object Reconstruction from a Single Depth View with Adversarial Learning](https://arxiv.org/abs/1708.07969) ([github](https://github.com/Yang7879/3D-RecGAN))
* ABC-GAN - [ABC-GAN: Adaptive Blur and Control for improved training stability of Generative Adversarial Networks](https://drive.google.com/file/d/0B3wEP_lEl0laVTdGcHE2VnRiMlE/view) ([github](https://github.com/IgorSusmelj/ABC-GAN))
* AC-GAN - [Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585) 
* acGAN - [Face Aging With Conditional Generative Adversarial Networks](https://arxiv.org/abs/1702.01983) 
* AdaGAN - [AdaGAN: Boosting Generative Models](https://arxiv.org/abs/1701.02386v1) 
* AE-GAN - [AE-GAN: adversarial eliminating with GAN](https://arxiv.org/abs/1707.05474) 
* AEGAN - [Learning Inverse Mapping by Autoencoder based Generative Adversarial Nets](https://arxiv.org/abs/1703.10094) 
* AffGAN - [Amortised MAP Inference for Image Super-resolution](https://arxiv.org/abs/1610.04490) 
* AL-CGAN - [Learning to Generate Images of Outdoor Scenes from Attributes and Semantic Layouts](https://arxiv.org/abs/1612.00215) 
* ALI - [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704) 
* AlignGAN - [AlignGAN: Learning to Align Cross-Domain Images with Conditional Generative Adversarial Networks](https://arxiv.org/abs/1707.01400) 
* AM-GAN - [Activation Maximization Generative Adversarial Nets](https://arxiv.org/abs/1703.02000) 
* AnoGAN - [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/abs/1703.05921v1) 
* ARAE - [Adversarially Regularized Autoencoders for Generating Discrete Structures](https://arxiv.org/abs/1706.04223) ([github](https://github.com/jakezhaojb/ARAE))
* ARDA - [Adversarial Representation Learning for Domain Adaptation](https://arxiv.org/abs/1707.01217) 
* ARIGAN - [ARIGAN: Synthetic Arabidopsis Plants using Generative Adversarial Network](https://arxiv.org/abs/1709.00938) 
* ArtGAN - [ArtGAN: Artwork Synthesis with Conditional Categorial GANs](https://arxiv.org/abs/1702.03410) 
* b-GAN - [Generative Adversarial Nets from a Density Ratio Estimation Perspective](https://arxiv.org/abs/1610.02920) 
* Bayesian GAN - [Deep and Hierarchical Implicit Models](https://arxiv.org/abs/1702.08896) 
* Bayesian GAN - [Bayesian GAN](https://arxiv.org/abs/1705.09558) 
* BCGAN - [Bayesian Conditional Generative Adverserial Networks](https://arxiv.org/abs/1706.05477) 
* BEGAN - [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717) 
* BGAN - [Binary Generative Adversarial Networks for Image Retrieval](https://arxiv.org/abs/1708.04150) ([github](https://github.com/htconquer/BGAN))
* BiGAN - [Adversarial Feature Learning](https://arxiv.org/abs/1605.09782v7) 
* BS-GAN - [Boundary-Seeking Generative Adversarial Networks](https://arxiv.org/abs/1702.08431v1) 
* C-RNN-GAN - [C-RNN-GAN: Continuous recurrent neural networks with adversarial training](https://arxiv.org/abs/1611.09904) ([github](https://github.com/olofmogren/c-rnn-gan/))
* CaloGAN - [CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks](https://arxiv.org/abs/1705.02355) ([github](https://github.com/hep-lbdl/CaloGAN))
* CAN - [CAN: Creative Adversarial Networks, Generating Art by Learning About Styles and Deviating from Style Norms](https://arxiv.org/abs/1706.07068) 
* CatGAN - [Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks](https://arxiv.org/abs/1511.06390v2) 
* CausalGAN - [CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training](https://arxiv.org/abs/1709.02023) 
* CC-GAN - [Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks](https://arxiv.org/abs/1611.06430) ([github](https://github.com/edenton/cc-gan))
* CDcGAN - [Simultaneously Color-Depth Super-Resolution with Conditional Generative Adversarial Network](https://arxiv.org/abs/1708.09105) 
* CGAN - [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) 
* CGAN - [Controllable Generative Adversarial Network](https://arxiv.org/abs/1708.00598) 
* Chekhov GAN - [An Online Learning Approach to Generative Adversarial Networks](https://arxiv.org/abs/1706.03269) 
* CoGAN - [Coupled Generative Adversarial Networks](https://arxiv.org/abs/1606.07536v2) 
* Conditional cycleGAN - [Conditional CycleGAN for Attribute Guided Face Image Generation](https://arxiv.org/abs/1705.09966) 
* constrast-GAN - [Generative Semantic Manipulation with Contrasting GAN](https://arxiv.org/abs/1708.00315) 
* Context-RNN-GAN - [Contextual RNN-GANs for Abstract Reasoning Diagram Generation](https://arxiv.org/abs/1609.09444) 
* Coulomb GAN - [Coulomb GANs: Provably Optimal Nash Equilibria via Potential Fields](https://arxiv.org/abs/1708.08819) 
* Cramèr GAN  - [The Cramer Distance as a Solution to Biased Wasserstein Gradients](https://arxiv.org/abs/1705.10743) 
* crVAE-GAN - [Channel-Recurrent Variational Autoencoders](https://arxiv.org/abs/1706.03729) 
* CS-GAN - [Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets](https://arxiv.org/abs/1703.04887) 
* CVAE-GAN - [CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training](https://arxiv.org/abs/1703.10155) 
* CycleGAN - [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) ([github](https://github.com/junyanz/CycleGAN))
* D2GAN - [Dual Discriminator Generative Adversarial Nets](http://arxiv.org/abs/1709.03831) 
* DAN - [Distributional Adversarial Networks](https://arxiv.org/abs/1706.09549) 
* DCGAN - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) ([github](https://github.com/Newmu/dcgan_code))
* DeliGAN - [DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data](https://arxiv.org/abs/1706.02071) ([github](https://github.com/val-iisc/deligan))
* DiscoGAN - [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192v1) 
* DistanceGAN - [One-Sided Unsupervised Domain Mapping](https://arxiv.org/abs/1706.00826) 
* DM-GAN - [Dual Motion GAN for Future-Flow Embedded Video Prediction](https://arxiv.org/abs/1708.00284) 
* DR-GAN - [Representation Learning by Rotating Your Faces](https://arxiv.org/abs/1705.11136) 
* DRAGAN - [How to Train Your DRAGAN](https://arxiv.org/abs/1705.07215) ([github](https://github.com/kodalinaveen3/DRAGAN))
* DSP-GAN - [Depth Structure Preserving Scene Image Generation](https://arxiv.org/abs/1706.00212) 
* DTN - [Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200) 
* DualGAN - [DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/abs/1704.02510v1) 
* Dualing GAN - [Dualing GANs](https://arxiv.org/abs/1706.06216) 
* EBGAN - [Energy-based Generative Adversarial Network](https://arxiv.org/abs/1609.03126v4) 
* ED//GAN - [Stabilizing Training of Generative Adversarial Networks through Regularization](https://arxiv.org/abs/1705.09367) 
* EGAN - [Enhanced Experience Replay Generation for Efficient Reinforcement Learning](https://arxiv.org/abs/1705.08245) 
* ExprGAN - [ExprGAN: Facial Expression Editing with Controllable Expression Intensity](https://arxiv.org/abs/1709.03842) 
* f-GAN - [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/abs/1606.00709) 
* FF-GAN - [Towards Large-Pose Face Frontalization in the Wild](https://arxiv.org/abs/1704.06244) 
* Fila-GAN - [Synthesizing Filamentary Structured Images with GANs](https://arxiv.org/abs/1706.02185) 
* Fisher GAN - [Fisher GAN](https://arxiv.org/abs/1705.09675) 
* Flow-GAN - [Flow-GAN: Bridging implicit and prescribed learning in generative models](https://arxiv.org/abs/1705.08868) 
* GAMN - [Generative Adversarial Mapping Networks](https://arxiv.org/abs/1709.09820) 
* GAN - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) ([github](https://github.com/goodfeli/adversarial))
* GAN-CLS - [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396) ([github](https://github.com/reedscot/icml2016))
* GAN-sep - [GANs for Biological Image Synthesis](https://arxiv.org/abs/1708.04692) ([github](https://github.com/aosokin/biogans))
* GAN-VFS - [Generative Adversarial Network-based Synthesis of Visible Faces from Polarimetric Thermal Faces](https://arxiv.org/abs/1708.02681) 
* GANCS - [Deep Generative Adversarial Networks for Compressed Sensing Automates MRI](https://arxiv.org/abs/1706.00051) 
* GAWWN - [Learning What and Where to Draw](https://arxiv.org/abs/1610.02454) ([github](https://github.com/reedscot/nips2016))
* GeneGAN - [GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data](https://arxiv.org/abs/1705.04932) ([github](https://github.com/Prinsphield/GeneGAN))
* Geometric GAN - [Geometric GAN](https://arxiv.org/abs/1705.02894) 
* GMAN - [Generative Multi-Adversarial Networks](http://arxiv.org/abs/1611.01673) 
* GMM-GAN - [Towards Understanding the Dynamics of Generative Adversarial Networks](https://arxiv.org/abs/1706.09884) 
* GoGAN - [Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking](https://arxiv.org/abs/1704.04865) 
* GP-GAN - [GP-GAN: Towards Realistic High-Resolution Image Blending](https://arxiv.org/abs/1703.07195) ([github](https://github.com/wuhuikai/GP-GAN))
* GRAN - [Generating images with recurrent adversarial networks](https://arxiv.org/abs/1602.05110) ([github](https://github.com/jiwoongim/GRAN))
* IAN - [Neural Photo Editing with Introspective Adversarial Networks](https://arxiv.org/abs/1609.07093) ([github](https://github.com/ajbrock/Neural-Photo-Editor))
* IcGAN - [Invertible Conditional GANs for image editing](https://arxiv.org/abs/1611.06355) ([github](https://github.com/Guim3/IcGAN))
* ID-CGAN - [Image De-raining Using a Conditional Generative Adversarial Network](https://arxiv.org/abs/1701.05957v3) 
* iGAN - [Generative Visual Manipulation on the Natural Image Manifold](https://arxiv.org/abs/1609.03552v2) ([github](https://github.com/junyanz/iGAN))
* Improved GAN - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) ([github](https://github.com/openai/improved-gan))
* InfoGAN - [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657v1) ([github](https://github.com/openai/InfoGAN))
* IRGAN - [IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval models](https://arxiv.org/abs/1705.10513v1) 
* IWGAN - [On Unifying Deep Generative Models](https://arxiv.org/abs/1706.00550) 
* l-GAN - [Representation Learning and Adversarial Generation of 3D Point Clouds](https://arxiv.org/abs/1707.02392) 
* LAGAN - [Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis](https://arxiv.org/abs/1701.05927) 
* LAPGAN - [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/abs/1506.05751) ([github](https://github.com/facebook/eyescream))
* LD-GAN - [Linear Discriminant Generative Adversarial Networks](https://arxiv.org/abs/1707.07831) 
* LDAN - [Label Denoising Adversarial Network (LDAN) for Inverse Lighting of Face Images](https://arxiv.org/abs/1709.01993) 
* LeakGAN - [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624) 
* LeGAN - [Likelihood Estimation for Generative Adversarial Networks](https://arxiv.org/abs/1707.07530) 
* LR-GAN - [LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation](https://arxiv.org/abs/1703.01560v1) 
* LS-GAN - [Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities](https://arxiv.org/abs/1701.06264) 
* LSGAN - [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076v3) 
* MAD-GAN - [Multi-Agent Diverse Generative Adversarial Networks](https://arxiv.org/abs/1704.02906) 
* MAGAN - [MAGAN: Margin Adaptation for Generative Adversarial Networks](https://arxiv.org/abs/1704.03817v1) 
* MalGAN - [Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN](https://arxiv.org/abs/1702.05983v1) 
* MaliGAN - [Maximum-Likelihood Augmented Discrete Generative Adversarial Networks](https://arxiv.org/abs/1702.07983) 
* MARTA-GAN - [Deep Unsupervised Representation Learning for Remote Sensing Images](https://arxiv.org/abs/1612.08879) 
* McGAN - [McGan: Mean and Covariance Feature Matching GAN](https://arxiv.org/abs/1702.08398v1) 
* MD-GAN - [Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks](https://arxiv.org/abs/1709.07592) 
* MDGAN - [Mode Regularized Generative Adversarial Networks](https://arxiv.org/abs/1612.02136) 
* MedGAN - [Generating Multi-label Discrete Electronic Health Records using Generative Adversarial Networks](https://arxiv.org/abs/1703.06490v1) 
* MGAN - [Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](https://arxiv.org/abs/1604.04382) ([github](https://github.com/chuanli11/MGANs))
* MGGAN - [Multi-Generator Generative Adversarial Nets](https://arxiv.org/abs/1708.02556) 
* MIX+GAN - [Generalization and Equilibrium in Generative Adversarial Nets (GANs)](https://arxiv.org/abs/1703.00573v3) 
* MMD-GAN - [MMD GAN: Towards Deeper Understanding of Moment Matching Network](https://arxiv.org/abs/1705.08584) ([github](https://github.com/dougalsutherland/opt-mmd))
* MMGAN - [MMGAN: Manifold Matching Generative Adversarial Network for Generating Images](https://arxiv.org/abs/1707.08273) 
* MoCoGAN - [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993) ([github](https://github.com/sergeytulyakov/mocogan))
* MPM-GAN - [Message Passing Multi-Agent GANs](https://arxiv.org/abs/1612.01294) 
* MuseGAN - [MuseGAN: Symbolic-domain Music Generation and Accompaniment with Multi-track Sequential Generative Adversarial Networks](https://arxiv.org/abs/1709.06298) 
* MV-BiGAN - [Multi-view Generative Adversarial Networks](https://arxiv.org/abs/1611.02019v1) 
* OptionGAN - [OptionGAN: Learning Joint Reward-Policy Options using Generative Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1709.06683) 
* ORGAN - [Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models ](https://arxiv.org/abs/1705.10843) 
* PAN - [Perceptual Adversarial Networks for Image-to-Image Transformation](https://arxiv.org/abs/1706.09138) 
* PassGAN - [PassGAN: A Deep Learning Approach for Password Guessing](https://arxiv.org/abs/1709.00440) 
* Perceptual GAN - [Perceptual Generative Adversarial Networks for Small Object Detection](https://arxiv.org/abs/1706.05274) 
* PGAN - [Probabilistic Generative Adversarial Networks](https://arxiv.org/abs/1708.01886) 
* pix2pix - [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) ([github](https://github.com/phillipi/pix2pix))
* PixelGAN - [PixelGAN Autoencoders](https://arxiv.org/abs/1706.00531) 
* Pose-GAN - [The Pose Knows: Video Forecasting by Generating Pose Futures](https://arxiv.org/abs/1705.00053) 
* PPGN - [Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space](https://arxiv.org/abs/1612.00005) 
* PrGAN - [3D Shape Induction from 2D Views of Multiple Objects](https://arxiv.org/abs/1612.05872) 
* PSGAN - [Learning Texture Manifolds with the Periodic Spatial GAN](http://arxiv.org/abs/1705.06566) 
* RankGAN - [Adversarial Ranking for Language Generation ](https://arxiv.org/abs/1705.11001) 
* RCGAN - [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633) 
* RefineGAN - [Compressed Sensing MRI Reconstruction with Cyclic Loss in Generative Adversarial Networks](https://arxiv.org/abs/1709.00753) 
* RenderGAN - [RenderGAN: Generating Realistic Labeled Data](https://arxiv.org/abs/1611.01331) 
* ResGAN - [Generative Adversarial Network based on Resnet for Conditional Image Restoration](https://arxiv.org/abs/1707.04881) 
* RNN-WGAN - [Language Generation with Recurrent Generative Adversarial Networks without Pre-training](https://arxiv.org/abs/1706.01399) ([github](https://github.com/liuyuemaicha/Adversarial-Learning-for-Neural-Dialogue-Generation-in-Tensorflow))
* RPGAN - [Stabilizing GAN Training with Multiple Random Projections](https://arxiv.org/abs/1705.07831) ([github](https://github.com/ayanc/rpgan))
* RTT-GAN - [Recurrent Topic-Transition GAN for Visual Paragraph Generation](https://arxiv.org/abs/1703.07022v2) 
* RWGAN - [Relaxed Wasserstein with Applications to GANs](https://arxiv.org/abs/1705.07164) 
* SAD-GAN - [SAD-GAN: Synthetic Autonomous Driving using Generative Adversarial Networks](https://arxiv.org/abs/1611.08788v1) 
* SalGAN - [SalGAN: Visual Saliency Prediction with Generative Adversarial Networks](https://arxiv.org/abs/1701.01081) ([github](https://github.com/imatge-upc/saliency-salgan-2017))
* SBADA-GAN - [From source to target and back: symmetric bi-directional adaptive GAN](https://arxiv.org/abs/1705.08824) 
* SD-GAN - [Semantically Decomposing the Latent Spaces of Generative Adversarial Networks](https://arxiv.org/abs/1705.07904) 
* SEGAN - [SEGAN: Speech Enhancement Generative Adversarial Network](https://arxiv.org/abs/1703.09452v1) 
* SeGAN - [SeGAN: Segmenting and Generating the Invisible](https://arxiv.org/abs/1703.10239) 
* SegAN - [SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation](https://arxiv.org/abs/1706.01805) 
* SeqGAN - [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473v5) ([github](https://github.com/LantaoYu/SeqGAN))
* SGAN - [Texture Synthesis with Spatial Generative Adversarial Networks](https://arxiv.org/abs/1611.08207) 
* SGAN - [Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.04357v4) ([github](https://github.com/xunhuang1995/SGAN))
* SGAN - [Steganographic Generative Adversarial Networks](https://arxiv.org/abs/1703.05502) 
* SimGAN - [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828) 
* SketchGAN - [Adversarial Training For Sketch Retrieval](https://arxiv.org/abs/1607.02748) 
* SL-GAN - [Semi-Latent GAN: Learning to generate and modify facial images from attributes](https://arxiv.org/abs/1704.02166) 
* SN-GAN - [Spectral Normalization for Generative Adversarial Networks](https://drive.google.com/file/d/0B8HZ50DPgR3eSVV6YlF3XzQxSjQ/view) ([github](https://github.com/pfnet-research/chainer-gan-lib))
* Softmax-GAN - [Softmax GAN](https://arxiv.org/abs/1704.06191) 
* Splitting GAN - [Class-Splitting Generative Adversarial Networks](https://arxiv.org/abs/1709.07359) 
* SRGAN - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) 
* SS-GAN - [Semi-supervised Conditional GANs](https://arxiv.org/abs/1708.05789) 
* ss-InfoGAN - [Guiding InfoGAN with Semi-Supervision](https://arxiv.org/abs/1707.04487) 
* SSGAN - [SSGAN: Secure Steganography Based on Generative Adversarial Networks](https://arxiv.org/abs/1707.01613) 
* SSL-GAN - [Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks](https://arxiv.org/abs/1611.06430v1) 
* ST-GAN - [Style Transfer Generative Adversarial Networks: Learning to Play Chess Differently](https://arxiv.org/abs/1702.06762) 
* StackGAN - [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.03242v1) 
* SteinGAN - [Learning Deep Energy Models: Contrastive Divergence vs. Amortized MLE](https://arxiv.org/abs/1707.00797) 
* S^2GAN - [Generative Image Modeling using Style and Structure Adversarial Networks](https://arxiv.org/abs/1603.05631v2) 
* TAC-GAN - [TAC-GAN - Text Conditioned Auxiliary Classifier Generative Adversarial Network](https://arxiv.org/abs/1703.06412v2) ([github](https://github.com/dashayushman/TAC-GAN))
* TAN - [Outline Colorization through Tandem Adversarial Networks](https://arxiv.org/abs/1704.08834) 
* TextureGAN - [TextureGAN: Controlling Deep Image Synthesis with Texture Patches](https://arxiv.org/abs/1706.02823) 
* TGAN - [Temporal Generative Adversarial Nets](https://arxiv.org/abs/1611.06624v1) 
* TP-GAN - [Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](https://arxiv.org/abs/1704.04086) 
* Triple-GAN - [Triple Generative Adversarial Nets](https://arxiv.org/abs/1703.02291v2) 
* Unrolled GAN - [Unrolled Generative Adversarial Networks](https://arxiv.org/abs/1611.02163) ([github](https://github.com/poolio/unrolled_gan))
* VAE-GAN - [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300) 
* VariGAN - [Multi-View Image Generation from a Single-View](https://arxiv.org/abs/1704.04886) 
* VAW-GAN - [Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/1704.00849) 
* VEEGAN - [VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning](https://arxiv.org/abs/1705.07761) ([github](https://github.com/akashgit/VEEGAN))
* VGAN - [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612) ([github](https://github.com/cvondrick/videogan))
* VGAN - [Generative Adversarial Networks as Variational Training of Energy Based Models](https://arxiv.org/abs/1611.01799) ([github](https://github.com/Shuangfei/vgan))
* ViGAN - [Image Generation and Editing with Variational Info Generative Adversarial Networks](https://arxiv.org/abs/1701.04568v1) 
* VIGAN - [VIGAN: Missing View Imputation with Generative Adversarial Networks](https://arxiv.org/abs/1708.06724) 
* VRAL - [Variance Regularizing Adversarial Learning](https://arxiv.org/abs/1707.00309) 
* WaterGAN - [WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images](https://arxiv.org/abs/1702.07392v1) 
* WGAN - [Wasserstein GAN](https://arxiv.org/abs/1701.07875v2) ([github](https://github.com/martinarjovsky/WassersteinGAN))
* WGAN-GP - [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) ([github](https://github.com/igul222/improved_wgan_training))
* WS-GAN - [Weakly Supervised Generative Adversarial Networks for 3D Reconstruction ](https://arxiv.org/abs/1705.10904) 
* α-GAN - [Variational Approaches for Auto-Encoding Generative Adversarial Networks](https://arxiv.org/abs/1706.04987) ([github](https://github.com/victor-shepardson/alpha-GAN))
* Δ-GAN - [Triangle Generative Adversarial Networks](https://arxiv.org/abs/1709.06548) 





