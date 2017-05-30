国内外从事[计算机视觉](http://lib.csdn.net/base/computervison)和图像处理相关领域的著名学者都以在三大顶级会议（ICCV，CVPR和ECCV）上发表论文为荣，其影响力远胜于一般SCI期刊论文，这三大顶级学术会议论文也引领着未来的研究趋势。CVPR是主要的计算机视觉会议，可以把它看作是计算机视觉研究的奥林匹克。博主今天先来整理**CVPR2015**年的精彩文章（这个就够很长一段时间消化的了）  
   顶级会议**CVPR2015**参会paper网址：  
[http://www.cv-foundation.org/openaccess/CVPR2015.py](http://www.cv-foundation.org/openaccess/CVPR2015.py)

来吧，一项项的开始整理，总有你需要的文章在等你！

## CNN Architectures {#cnn-architectures}

CNN网络结构：  
1.Hypercolumns for Object Segmentation and Fine-Grained Localization  
Authors: Bharath Hariharan, Pablo Arbeláez, Ross Girshick, Jitendra Malik

2.Modeling Local and Global Deformations in Deep Learning: Epitomic Convolution, Multiple Instance Learning, and Sliding Window Detection  
Authors: George Papandreou, Iasonas Kokkinos, Pierre-André Savalle

3.Going Deeper With Convolutions  
Authors: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich  
这篇文章推荐一下，使用了**《network in network》**中的用 global averaging pooling layer 替代 fully-connected layer的思想。有看过的可以私信博主，一起讨论文章心得。

4.Improving Object Detection With Deep Convolutional Networks via Bayesian Optimization and Structured Prediction  
Authors: Yuting Zhang, Kihyuk Sohn, Ruben Villegas, Gang Pan, Honglak Lee

5.Deep Neural Networks Are Easily Fooled: High Confidence Predictions for Unrecognizable Images  
Authors: Anh Nguyen, Jason Yosinski, Jeff Clune

## Action and Event Recognition {#action-and-event-recognition}

1.Deeply Learned Attributes for Crowded Scene Understanding  
Authors: Jing Shao, Kai Kang, Chen Change Loy, Xiaogang Wang

2.Modeling Video Evolution for Action Recognition  
Authors: Basura Fernando, Efstratios Gavves, José Oramas M., Amir Ghodrati, Tinne Tuytelaars

3.Joint Inference of Groups, Events and Human Roles in Aerial Videos  
Authors: Tianmin Shu, Dan Xie, Brandon Rothrock, Sinisa Todorovic, Song Chun Zhu

## Segmentation in Images and Video {#segmentation-in-images-and-video}

1.Causal Video Object Segmentation From Persistence of Occlusions  
Authors: Brian Taylor, Vasiliy Karasev, Stefano Soatto

2.**Fully Convolutional Networks for Semantic Segmentation**  
Authors: Jonathan Long, Evan Shelhamer, Trevor Darrell  
——文章把全连接层当做卷积层，也用来输出featuremap。这样相比Hypercolumns/HED 这样的模型，可迁移的模型层数（指VGG16/Alexnet等）就更多了。但是从文章来看，因为纯卷积嘛，所以featuremap的每个点之间没有位置信息的区分。相较于Hypercolumns的claim，鼻子的点出现在图像的上半部分可以划分为pedestrian类的像素，但是如果出现在下方就应该划分为背景。所以位置信息应该是挺重要需要考虑的。这也许是速度与性能的trade-off?

3.Is object localization for free - Weakly-supervised learning with convolutional neural networks  
——弱监督做object detection的文章。首先fc layer当做conv layer与上面这篇文章思想一致。同时把最后max pooling之前的feature map看做包含class localization的信息，只不过从第五章“Does adding object-level supervision help classification”的结果看，效果虽好，但是这一物理解释可能不够完善。

4.Shape-Tailored Local Descriptors and Their Application to Segmentation and**Tracking**  
Authors: Naeemullah Khan, Marei Algarni, Anthony Yezzi, Ganesh Sundaramoorthi

5.**Deep Filter Banks for Texture Recognition and Segmentation**  
Authors: Mircea Cimpoi, Subhransu Maji, Andrea Vedaldi

6.Deeply learned face representations are sparse, selective, and robust, Yi Sun, Xiaogang Wang, Xiaoou Tang  
——DeepID系列之DeepID2+。在DeepID2之上的改进是增加了网络的规模\(feature map数目\)，另外每一层都接入一个全连通层加supervision。最精彩的地方应该是后面对神经元性能的分析，发现了三个特点：1.中度稀疏最大化了区分性，并适合二值化；2.身份和attribute选择性；3.对遮挡的鲁棒性。这三个特点在模型训练时都没有显示或隐含地强加了约束，都是CNN自己学的。

## Image and Video Processing and Restoration {#image-and-video-processing-and-restoration}

1.Fast and Flexible Convolutional Sparse Coding  
Authors: Felix Heide, Wolfgang Heidrich, Gordon Wetzstein

2.What do 15,000 Object Categories Tell Us About Classifying and Localizing Actions?  
Authors: Mihir Jain, Jan C. van Gemert, Cees G. M. Snoek  
——物品的分类对行为检测有帮助作用。这篇文章是第一篇关于这个话题进行探讨的，是个深坑，大家可以关注一下，考虑占坑。

3.**Hypercolumns for Object Segmentation and Fine-Grained Localization**  
Authors:Bharath Hariharan, Pablo Arbeláez, Ross Girshick, Jitendra Malik  
——一个很好的思路！以前的CNN或者R-CNN，我们总是用最后一层作为class label，倒数第二层作为feature。这篇文章的作者想到利用每一层的信息。因为对于每一个pixel来讲，在所有层数上它都有被激发和不被激发两种态，作者利用了每一层的激发态作为一个feature vector来帮助自己做精细的物体检测。

## 3D Models and Images {#3d-models-and-images}

1.The Stitched Puppet: A Graphical Model of 3D Human Shape and Pose  
Authors: Silvia Zuffi, Michael J. Black

2.3D Shape Estimation From 2D Landmarks: A Convex Relaxation Approach  
Authors: Xiaowei Zhou, Spyridon Leonardos, Xiaoyan Hu, Kostas Daniilidis

## Images and Language {#images-and-language}

这个类别的文章需要好好看看，对思路的发散很有帮助

1.**Show and Tell: A Neural Image Caption Generator**  
Authors: Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan

2.**Deep Visual-Semantic Alignments for Generating Image Descriptions**  
Authors: Andrej Karpathy, Li Fei-Fei

3.**Long-Term Recurrent Convolutional Networks for Visual Recognition and Description**  
Authors: Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, Trevor Darrell

4.Becoming the Expert - Interactive Multi-Class Machine Teaching  
Authors: Edward Johns, Oisin Mac Aodha, Gabriel J. Brostow

## 其它 {#其它}

参考文献一：CNN卷积神经网络的改进（15年最新paper）：  
[http://blog.csdn.net/u010402786/article/details/50499864](http://blog.csdn.net/u010402786/article/details/50499864)  
文章中的四篇文章也值得一读，其中一篇在上面出现过。一定要自己下载下来看一看。  
参考文献二：这是另外一个博主的博客，也是对CVPR的文章进行了整理：  
[http://blog.csdn.net/jwh\_bupt/article/details/46916653](http://blog.csdn.net/jwh_bupt/article/details/46916653)



今天要跟大家分享的是一些 Convolutional Neural Networks（CNN）的工作。大家都知道，CNN 最早提出时，是以一定的人眼生理结构为基础，然后逐渐定下来了一些经典的[架构](http://lib.csdn.net/base/architecture)——convolutional 和 pooling 的交替，最后再加上几个 fully-connected layers 用作最后做 prediction 等的输出。然而，如果我们能“反思”经典，深入剖析这些经典架构中的不同 component 的作用，甚至去改进它们，有时候可能有更多的发现。所以，今天分享的内容，便是改进 CNN 的一些工作。

## Striving For Simplicity：The All Convolutional Net {#striving-for-simplicitythe-all-convolutional-net}

先说一篇探究 CNN 中不同 component 的重要性和作用的工作。这篇工作发表于 ICLR 2015，已经有一年多的时间了。个人觉得它应该受到更大的关注。这篇工作最大的贡献是，把经典架构的 CNN 中的 pooling 层，用 stride convolutional 层给替换掉了，也就是去掉了 deterministic 层。并且，通过数学公式和实验结果证明，这样的替换是完全不会损伤性能的。而加入 pooling 层，如果不慎，甚至是有伤性能的。

具体来看，CNN 中的每次 feature map 表达，都可以看做一个 W\*H\*N 的三维结构。在这个三维结构下，pooling 这种 subsampling 的操作，可以看成是一个用 p-norm（当 p 趋向于正无穷时，就是我们最常见的 max-pooling）当做 activation function 的 convolutional 操作。有了这样的解释，自然而然地，就会提出一个问题：那么引入这样的 pooling 操作，有什么意义呢？真的有必要么？在过去的工作中，大家普遍认为，pooling 层有三种可能的功效：（1）提取更 invariant 的 feature；（2）抽取更广范围的（global）的 feature——即 spatially dimension reduction；（3）更方便优化。这篇作者认为，其中（2）是对 CNN 最重要的。基于此，它们就提出，那么我只要在去掉 pooling 层的同时，保证这种 spatially dimension reduction——是否就可以构造出一个**all convolutional net**，同时这个 net 的效果还不比 convolutional + pooling 交替的差？  
![](http://img.blog.csdn.net/20160111214701048 "这里写图片描述")

最后，果然，如他们所料。它们的 all convolutional net 达到了甚至有时候超过了 state-of-art，同时他们还发现有时候加入 pooling 反而不如不加。这篇工作的另外一个贡献是他们提出了一种新的 visualizing CNN 的方法，效果更直观更有分辨性。总结来说，这篇工作提出的 all convolutional net 可以实现自主 downsampling，并且效果不差；其二，它不是在说 pooling 不好，我们一定要抛弃；而是对于 minimum necessary ingredients for CNN 进行了探究。

## Network in Network {#network-in-network}

第一个要分享的是如何去 replace pooling layer after convolutional layer，接下来要分享的是 replace fully-connected layer on the top of CNN。这个也是经典的 CNN 架构中的一个组成部分。也许这个东西大家还不熟悉，但是提到另一个词，大家就会很熟悉了。那就是，dropout。已经有非常多的工作，在 CNN 的 fully-connected layer 中，加入 dropout，来避免 overfitting。受此启发，后来又有了一个 sparse convolutional neural networks 的工作。然而，更具开创性的工作是，《Network in Network》这篇，提出了用**global averaging pooling layer**替代 fully-connected layer.  
![](http://img.blog.csdn.net/20160111214748711 "这里写图片描述")  
这样的 global averaging pooling layer 显然，可以把 input/feature map 和 output/category，也就可以达到减少 overfitting 的作用（和 dropout 一样）。  
![](http://img.blog.csdn.net/20160111214803165 "这里写图片描述")  
现在，global average pooling 已经被用得很广泛，比如在《Going Deeper with Convolutions》中，作者指出：  
We found that a move from fully connected  
layers to average pooling improved the top-1 accuracy by  
about 0.6%, however the use of dropout remained essential  
even after removing the fully connected layers.  
当然它也有它自己的一定弊端，比如 convergence 会变慢。但是关于 fully-connected layer 是否一定是必须的这个问题，却在被各种工作和各种场景探究。比如 Google 去年非常火的 inceptionism 的工作中，也放弃了 fully-connected layer，并达到了很好的效果。  
所以，关于这点，小S 是认为，我们不要光看表面的 dropout or global averaging pooling 这些技术，而是要去思考它们的共同之处和它们的原理。从它们带给网络结构的变化入手。也许现在来看，最初的结论还是对的，deeper is better，我们暂时要解决的是如何 deeper。

## Spatial Transformer Networks {#spatial-transformer-networks}

这篇是 NIPS 2015 中，来自 Google DeepMind 的工作。这篇也被前几天 huho larochelle 评选出的 Top 10 arXiv 2015 Deep Learning Papers 收录（另外提一下，昨天看到这个评选，发现大部分我都写过笔记了，大家如果感兴趣，我可以单独整理一份，以供大家查阅）。回到这篇工作上来，它主要是说，尽管 CNN 一直号称可以做 spatial invariant feature extraction，但是这种 invariant 是很有局限性的。因为 CNN 的 max-pooling 首先只是在一个非常小的、rigid 的范围内（2×2 pixels）进行，其次即使是 stacked 以后，也需要非常 deep 才可以得到大一点范围的 invariant feature，三者来说，相比 attention 那种只能抽取 relevant 的 feature，我们需要的是更广范围的、更 canonical 的 features。为此它们提出了一种新的完全 self-contained transformation module，可以加入在网络中的任何地方，灵活高效地提取 invariant image features.  
![](http://img.blog.csdn.net/20160111214823618 "这里写图片描述")  
具体上，这个 module 就叫做 Spatial Transformers，由三个部分组成： Localization Network, Grid generator 和 Sampler。Localization Network 非常灵活，可以认为是一个非常 general 的进一步生成 feature map 和 map 对应的 parameter 的网络。因此，它不局限于用某一种特定的 network，但是它要求在 network 最后有一层 regression，因为需要将 feature map 的 parameter 输出到下一个部分：Grid generator。Grid generator 可以说是 Spatial Transformers 的核心，它主要就是生成一种“蒙版”，用于“抠图”（Photoshop 附体……）。Grid generator 定义了 Transformer function，这个 function 的决定了能不能提取好 invariant features。如果是 regular grid，就好像一张四四方方没有倾斜的蒙版，是 affined grid，就可以把蒙版“扭曲”变换，从而提取出和这个蒙版“变换”一致的特征。在这个工作中，只需要六个参数就可以把 cropping, translation, rotation, scale and skew 这几种 transformation 都涵盖进去，还是很强大的；而最后的 Sampler 就很好理解了，就是用于把“图”抠出来。  
![](http://img.blog.csdn.net/20160111214839543 "这里写图片描述")  
这个工作有非常多的优点：（1）它是 self-contained module，可以加在网络中的任何地方，加任何数量，不需要改变原网络；（2）它是 differentiable 的，所以可以直接进行各种 end-to-end 的训练；（3）它这个 differentiable simple and fast，所以不会使得原有网络变慢；（4）相比于 pooling 和 attention 机制，它抽取出的 invariant features 更 general。

## Stacked What-Where Auto-encoders {#stacked-what-where-auto-encoders}

这篇文章来自 NYU，Yann LeCun 组，已投稿到 ICLR 2016。与之前整理过的 improving information flow in Seq2Seq between encoder-decoder 类似的是，这篇文章主要是改进了基于 CNN 的 encoder-decoder，并非常 intuitive 的讨论了不同 regularizer 的区别。架构图可以直接看 Figure 1 的右侧，会比较清晰。具体来讲，**Stacked What-Where Auto-encoders（SWWAE） 基于前向 Convnet 和前向 Deconvnet**，并将 max-pooling 的输出称为 “what”，其实就是将 max function 的 content 和 position 传给下一层；同时，max-pooling 中的 position/location 信息，也就是 argmax function，作为 “where” 要“横向”传给 decoder。这样，在进行 decoder reconstruct 的过程时，则更能基于 where + what 的组合，进行 unpooling。  
为了能让网络利用好 what 和 where，文章考虑了三种 loss，见公式（1），即传统的 discriminate loss，和新增的 input-level reconstruction loss for “what” 还有 intermediate-level reconstruction loss for “where”。  
![](http://img.blog.csdn.net/20160111215029452 "这里写图片描述")

如上文所说，文章的 Section 3 很 intuitive，首先说明并解了为什么使用的是 soft version 的 max/argmax 去进行 ”what“ 和 ”where“；第二，讨论了为何加入 reconstruction loss 和这样一个 hybird loss function 更好（generalization 和 robustness\)；第三，说明了 intermediate loss 对于“what”“where”一起学习的重要性。  
实验结果上来看，这样的 SWWAE 模型 generate 出来的图片更清晰，更“干净”（clearer and cleaner）。

