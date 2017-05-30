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

