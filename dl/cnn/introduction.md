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

