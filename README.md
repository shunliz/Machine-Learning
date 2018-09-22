# 机器学习原理

---

机器学习原理笔记整理. Gitbook地址[https://shunliz.gitbooks.io/machine-learning/content/](https://shunliz.gitbooks.io/machine-learning/content/)  
前半部分关注数学基础，机器学习和深度学习的理论部分，详尽的公式推导。  
后半部分关注工程实践和理论应用部分

[如何贡献](/contribute.md)？

# 赞助

如果您觉得这个资料还不错，您也可以打赏一下。

![](/assets/mywebchat.png)

---

* [前言](README.md)
* [第一部分 数学基础](math/math.md)
* [第一章 数学分析](math/analytic/introduction.md)
  * [梯度下降](math/analytic/gradient_descent.md)
  * [数值计算](math/analytic/shu-zhi-ji-suan.md)
  * [过拟合数学原理与解决方案](math/analytic/overfitting.md)
  * [交叉验证](math/analytic/cross-validation.md)
  * [最小二乘](math/analytic/least-square.md)
  * [拉格朗日乘子法（Lagrange Multiplier\) 和KKT条件](math/analytic/lagelangri-kkt.md)
  * [牛顿法](math/analytic/niudun.md)
  * [凸优化](math/analytic/tuyouhua.md)
  * [常用函数](math/analytic/common-function.md)
* [第二章 概率论](math/probability.md)
  * [统计学习方法概论](math/probability/prob-methodology.md)
  * [最大似然估计](math/probability/mle.md)
  * [蒙特卡罗方法](math/probability/mcmc1.md)
  * [马尔科夫链](math/probability/markov-chain.md)
  * [MCMC采样和M-H采样](math/probability/mcmc-mh.md)
  * [Gibbs采样](math/probability/gibbs.md)
* [第三章 矩阵和线性代数](math/linear-matrix/linear-matrix.md)
  * [LAPACK](math/linear-matrix/lapack.md)
  * [特征值与特征向量](math/linear-matrix/tezhengzhihetezhengxiangliang.md)
* [第二部分 机器学习](ml/ml.md)
* [第四章 机器学习基础](ml/pythonml.md)
  * [Python及其数学库](ml/pythonml/pythonji-qi-shu-xue-ku.md)
  * [机器学习库](ml/pythonml/ji-qi-xue-xi-ku.md)
  * [模型度量](ml/pythonml/ml-metrics.md)
  * [生成模型和判别模型](ml/pythonml/gen-descri.md)
  * [机器学习中的距离](ml/pythonml/distance.md)
* [第六课：数据清洗和特征选择](ml/clean-feature/cleanup-feature.md)
  * [PCA](ml/clean-feature/pca.md)
  * [ICA](ml/clean-feature/ica.md)
  * [One-hot编码](ml/clean-feature/one-hot.md)
  * [scikit-learn PCA](ml/clean-feature/scikit-pca.md)
  * [线性判别分析LDA](ml/clean-feature/xian-xing-pan-bie-fen-xi-lda.md)
  * [用scikit-learn进行LDA降维](ml/clean-feature/scikit-lda.md)
  * [奇异值分解\(SVD\)原理与在降维中的应用](ml/clean-feature/svd.md)
  * [局部线性嵌入\(LLE\)原理](ml/clean-feature/lle.md)
  * [scikit-learn LLE](ml/clean-feature/scikit-lle.md)
  * [spark特征选择](ml/clean-feature/spark-fselect.md)
  * [Spark特征提取](ml/clean-feature/spark-fextract.md)
  * [异常数据监测](ml/clean-feature/outlier-detect.md)
  * [数据预处理](ml/clean-feature/datapreprocess.md)
  * [特征工程](ml/clean-feature/te-zheng-gong-cheng.md)
* [第七课： 回归](ml/regression/regression.md)
  * [1.  线性回归](ml/regression/linear-regression.md)
  * [10.最大熵模型](ml/regression/max-entropy.md)
  * [11.K-L散度](ml/regression/kl.md)
  * [坐标下降和最小角](ml/regression/cordinate-angle.md)
  * [线性回归小结](ml/regression/linear-regression-summary.md)
  * [Logistic回归](ml/regression/logistic.md)
  * [Logistic回归小结](ml/regression/logistichui-gui-xiao-jie.md)
  * [SoftMax回归](ml/regression/softmax.md)
* [第九课：决策树](ml/decisiontree.md)
  * [ID3](ml/decisiontree/id3.md)
  * [C4.5](ml/decisiontree/c45.md)
  * [CART](ml/decisiontree/cart.md)
  * [总结](ml/decisiontree/summary.md)
  * [实现代码](ml/decisiontree/code.md)
* [第十三课：SVM](ml/svm.md)
  * [感知机模型](ml/svm/gan-zhi-ji-mo-xing.md)
  * [线性SVM](ml/svm/linear-svm.md)
  * [软间隔最大化模型](ml/svm/soft-margin-max.md)
  * [核函数](ml/svm/kernel-method.md)
  * [SMO算法原理](ml/svm/smo.md)
  * [SVM回归](ml/svm/svm-regression.md)
  * [scikit-learn SVM](ml/svm/scikit-learn-svm.md)
  * [支持向量机高斯核调参](ml/svm/gaosi-kernel.md)
  * [SVM代码实现](ml/svm/svm-code.md)
* [集成学习](ml/integrate.md)
  * [Adaboost原理](ml/integrate/adaboost.md)
  * [scikit-learn Adaboost](ml/integrate/scikit-learn-adaboost.md)
  * [梯度提升树（GBDT）](ml/integrate/gbdt.md)
  * [scikit GBDT](ml/integrate/scikit-gbdt.md)
  * [Bagging与随机森林](ml/integrate/random-forest.md)
  * [XGBOOST](ml/integrate/xgboost.md)
  * [scikit-learn 随机森林](ml/integrate/scikit-learn-rf.md)
* [第十五课：聚类](ml/cluster.md)
  * [K-Mean](ml/cluster/kmeans.md)
  * [KNN](ml/cluster/KNN.md)
  * [scikit-learn KNN](ml/cluster/knnshi-jian.md)
  * [KNN 代码](ml/cluster/knn-code.md)
  * [scikit-learn K-Means](ml/cluster/scikit-k-means.md)
  * [BIRCH聚类算法原理](ml/cluster/birch.md)
  * [scikit-learn BIRCH](ml/cluster/scikit-learn-birch.md)
  * [DBSCAN密度聚类算法](ml/cluster/dbscan.md)
  * [scikit-learn DBSCAN](ml/cluster/scikit-learn-dbscan.md)
  * [谱聚类（spectral clustering）原理](ml/cluster/spectral.md)
  * [scikit-learn 谱聚类](ml/cluster/scikit-spectral.md)
  * [近邻传播算法](ml/cluster/ap.md)
  * [混合高斯模型](ml/cluster/gmm.md)
* [关联分析](ml/associative/associative.md)
  * [典型关联分析\(CCA\)原理](ml/associative/cca.md)
  * [Apriori算法原理](ml/associative/apriori.md)
  * [FP Tree算法原理](ml/associative/fptree.md)
  * [PrefixSpan算法原理](ml/associative/prefixspan.md)
  * [Spark FP Tree算法和PrefixSpan算法](ml/associative/spark-fptree-prefixspan.md)
* [推荐算法](ml/recommand/recommand.md)
  * [矩阵分解协同过滤推荐算法](ml/recommand/matrix-filter.md)
  * [SimRank协同过滤推荐算法](ml/recommand/simrank.md)
  * [Spark矩阵分解推荐算法](ml/recomand/spark-factor.md)
  * [分解机\(Factorization Machines\)推荐算法原理](ml/recommand/fm.md)
  * [美团推荐算法](ml/recommand/meituan.md)
  * [MapReduce ItemCF](ml/recommand/mr-itemcf.md)
  * [基于标签的用户推荐系统](ml/recommand/label-recommand.md)
* [第十七课：EM算法](ml/em/em.md)
* [第十九课：贝叶斯网络](ml/bayes.md)
  * [朴素贝叶斯](ml/bayes/po-su-bei-xie-si.md)
  * [scikit-learn朴素贝叶斯](ml/bayes/scikit-simple-bayes.md)
  * [朴素贝叶斯实际应用](ml/bayes/simple-bayes-real-use.md)
  * [朴素贝叶斯代码](ml/bayes/simple-bayes-code.md)
* [第二十一课：LDA主题模型](ml/lda/lda.md)
* [第二十三课：隐马尔科夫模型HMM](ml/hmm/hmm.md)
  * [HMM前向后向算法评估观察序列概率](ml/hmm/hmm-forward-backward.md)
  * [鲍姆-韦尔奇算法求解HMM参数](ml/hmm/bmwl-hmm.md)
  * [维特比算法解码隐藏状态序列](ml/hmm/viterb-hmm.md)
  * [用hmmlearn学习隐马尔科夫模型HMM](ml/hmm/hmmlearn.md)
  * [马尔科夫蒙特卡洛](ml/hmm/markv-mengtekluo.md)
* [条件随机场CRF](ml/crf/crf.md)
  * [从随机场到线性链条件随机场](ml/crf/linear-crf.md)
  * [前向后向算法评估标记序列概率](ml/crf/back-forth.md)
  * [维特比算法解码](ml/crf/crf-viterbi.md)
* [第三部分 深度学习](dl/dl.md)
* [深度学习层](dl/layers/layers.md)
  * [核心层](dl/layers/core.md)
  * [卷积层](dl/layers/conv.md)
  * [池化层](dl/layers/pooling.md)
  * [局部连接层](dl/layers/lcnn.md)
  * [循环层](dl/layers/rnn.md)
  * [嵌入层](dl/layers/ebbedded.md)
  * [合并层](dl/layers/merge.md)
  * [高级激活层](dl/layers/activation.md)
  * [归一化层](dl/layers/regular.md)
  * [噪声层](dl/layers/nosie.md)
  * [层包裹](dl/layers/wrapper.md)
  * [自定义层](dl/layers/userdefine.md)
* [第二十五课：深度学习](dl/introduction/introduction.md)
  * [基本概念](dl/introduction/ji-ben-gai-nian.md)
  * [深度神经网络（DNN）模型与前向传播算法](dl/introduction/dnn-fp.md)
  * [深度神经网络（DNN）反向传播算法\(BP\)](dl/introduction/dnn-bp.md)
  * [反向传播](dl/introduction/back-propagation.md)
  * [反向传播2](dl/introduction/READ.md)
  * [DNN损失函数和激活函数的选择](dl/introduction/dnn-loss.md)
  * [深度神经网络（DNN）的正则化](dl/introduction/dnn-normal.md)
  * [参考文献](dl/reference.md)
* [第二十六课 卷积神 经网络\(Convolutional Neural Netowrk\)](dl/cnn/introduction.md)
  * [卷积神经网络\(CNN\)模型结构](dl/cnn/cnn-arch.md)
  * [卷积神经网络\(CNN\)前向传播算法](dl/cnn/cnn-fp.md)
  * [卷积神经网络\(CNN\)反向传播算法](dl/cnn/cnn-bp.md)
* [对抗生成网络\(Generative Adversarial Networks\)](dl/gan/gan.md)
  * [GAN原理](ml/gan/gan-principle.md)
  * [InfoGAN](dl/gan/infogan.md)
  * [DCGAN](dl/gan/dcgan.md)
  * [VAE](dl/gan/vae.md)
* [受限波尔兹曼机](dl/rbm/rbm.md)
  * [RBM code](dl/rbm/rbm-code.md)
  * [DBN](dl/rbm/dbn.md)
  * [RBM原理](dl/rbm/rbm-yuanli.md)
* [RNN](dl/rnn/rnn.md)
  * [Bidirectional RNNs](dl/rnn/bidirectional-rnns.md)
  * [Deep \(Bidirectional\) RNNs](dl/rnn/deep-bidirectional-rnns.md)
  * [LSTM模型与前向反向传播算法](dl/rnn/lstm.md)
  * [随时间反向传播（BPTT）算法](dl/rnn/bptt.md)
  * [循环神经网络\(RNN\)模型与前向反向传播算法](dl/rnn/rnn-bptt.md)
* [自动编码器](dl/encoder/encoder.md)
  * [堆叠降噪自动编码器](dl/encoder/stack-denoise-encoder.md)
  * [降噪自动编码器](dl/encoder/denoise-encoder.md)
  * [sparse自动编码器](dl/encoder/sparse-autoencoder.md)
  * [Keras自动编码器](dl/encoder/keras-autoencoder.md)
* [word2vec](dl/word2vec/word2vec.md)
  * [CBOW与Skip-Gram模型基础](dl/word2vec/cbow-skip-n.md)
  * [基于Hierarchical Softmax的模型](dl/word2vec/hierarc-softmax.md)
  * [基于Negative Sampling的模型](dl/word2vec/negative-sampling.md)
* [增强学习](dl/reinforcement/reinforcement.md)
  * [Q-Learning](dl/reinforcement/q-learning.md)
  * [策略网络](dl/reinforcement/policy-network.md)
  * [bandit算法](dl/reinforcement/banditsuan-fa.md)
  * [蒙特卡洛树搜索](dl/reinforcement/meng-te-qia-luo-shu-sou-suo.md)
  * [多臂赌博机\(Multi-arm Bandits\)](dl/reinforcement/multi-bandit.md)
  * [马尔可夫决策过程MDP](dl/reinforcement/mdp.md)
  * [动态编程](dl/reinforcement/dynamic-programming.md)
  * [蒙特卡洛方法](dl/reinforcement/monte-carlo.md)
  * [时序差分学习](dl/reinforcement/shi-xu-cha-fen-xue-xi.md)
  * [A3C算法](dl/reinforcement/a3csuan-fa.md)
  * [Multi-steps bootstraping](dl/reinforcement/multi-steps-bootstraping.md)
  * [Planning and Learning with Tabular Methods](dl/reinforcement/tabular-method.md)
  * [DQN](dl/reinforcement/dqn.md)
  * [Policy Gridient](dl/reinforcement/policy-gridient.md)
  * [Actor Critic](dl/reinforcement/actor-critic.md)
  * [DDPG \(Deep Deterministic Policy Gradient\)](dl/reinforcement/ddpg.md)
  * [PPO\(Proximal Policy Optimization \)](dl/reinforcement/ppo.md)
  * [Alpha-Beta剪枝算法详解](dl/reinforcement/alpha-beta.md)
* [进化算法](ml/evolution/evolution.md)
  * [遗传算法](ml/evolution/yichuansuanfa.md)
  * [进化策略](ml/evolution/evolution-strategy.md)
  * [NEAT](ml/evolution/neat.md)
* [自然语言处理](nlp/nlp.md)
  * [文本挖掘的分词原理](nlp/text-mine.md)
  * [HashTrick](nlp/hashtrick.md)
  * [TF-IDF](nlp/tf-idf.md)
  * [中文文本挖掘预处理](nlp/preprocessing.md)
  * [英文文本挖掘预处理](nlp/english-preprocess.md)
  * [潜在语义索引\(LSI\)](nlp/lda/lsi.md)
  * [非负矩阵分解\(NMF\)](nlp/lda/nmf.md)
  * [LDA基础](nlp/lda/lda.md)
  * [LDA求解之Gibbs采样算法](nlp/lda/lda-gibbs.md)
  * [LDA求解之变分推断EM算法](nlp/lda/vi-em.md)
  * [scikit-learn LDA主题模型](nlp/lda/scikit-learn-lda.md)
* [语音识别](voice/introduction.md)
  * [GMM-HMM](voice/gmm-hmm.md)
  * [目录](voice/voicemulumd.md)
* [第四部分 学习资源](resources/introduction.md)
  * [机器学习](resources/ml/list.md)
  * [强化学习](resources/rl/list.md)
  * [自然语言处理](resources/nlp/list.md)
  * [深度学习](resources/dl/list.md)
* 流行网络结构
  * [mobilenet](dl/popnet/mobilenet.md)
  * [ResNet](dl/popnet/resnet.md)
* 并行学习
  * [美团并行学习实践](dl/paralleldl/mei-tuan-bing-xing-xue-xi-shi-jian.md)
* AI应用
  * [美团外卖AI技术](dl/dlapp/mei-tuan-wai-mai-ai-ji-zhu.md)
  * [美团推荐排序](dl/dlapp/mei-tuan-tui-jian-pai-xu.md)

---

![](/images/machine-learning-classification.jpg)

机器学习无疑是当前数据分析领域的一个热点内容。很多人在平时的工作中都或多或少会用到机器学习的算法。

机器学习的算法很多。很多时候困惑人们都是，很多算法是一类算法，而有些算法又是从其他算法中延伸出来的。这里，我们从两个方面来给大家介绍，第一个方面是学习的方式，第二个方面是算法的类似性。

## **学习方式**

根据数据类型的不同，对一个问题的建模有不同的方式。在机器学习或者人工智能领域，人们首先会考虑算法的学习方式。在机器学习领域，有几种主要的学习方式。将算法按照学习方式分类是一个不错的想法，这样可以让人们在建模和算法选择的时候考虑能根据输入数据来选择最合适的算法来获得最好的结果。

### **监督式学习：**

![](/assets/mlpre1.png)

在监督式学习下，输入数据被称为“训练数据”，每组训练数据有一个明确的标识或结果，如对防垃圾邮件系统中“垃圾邮件”“非垃圾邮件”，对手写数字识别中的“1“，”2“，”3“，”4“等。在建立预测模型的时候，监督式学习建立一个学习过程，将预测结果与“训练数据”的实际结果进行比较，不断的调整预测模型，直到模型的预测结果达到一个预期的准确率。监督式学习的常见应用场景如分类问题和回

归问题。常见算法有逻辑回归（Logistic Regression）和反向传递神经网络（Back Propagation Neural Network）

### **非监督式学习：**

![](/assets/mlpre2.png)

在非监督式学习中，数据并不被特别标识，学习模型是为了推断出数据的一些内在结构。常见的应用场景包括关联规则的学习以及聚类等。常见算法包括Apriori算法以及K-Means算法。

### **半监督式学习：**

![](/assets/mlpre3.png)

在此学习方式下，输入数据部分被标识，部分没有被标识，这种学习模型可以用来进行预测，但是模型首先需要学习数据的内在结构以便合理的组织数据来进行预测。应用场景包括分类和回归，算法包括一些对常用监督式学习算法的延伸，这些算法首先试图对未标识数据进行建模，在此基础上再对标识的数据进行预测。如图论推理算法（Graph Inference）或者拉普拉斯支持向量机（Laplacian SVM.）等。

### **强化学习：**

![](/assets/mlpre4.png)

在这种学习模式下，输入数据作为对模型的反馈，不像监督模型那样，输入数据仅仅是作为一个检查模型对错的方式，在强化学习下，输入数据直接反馈到模型，模型必须对此立刻作出调整。常见的应用场景包括动态系统以及[机器人](http://lib.csdn.net/base/robot)控制等。常见算法包括Q-Learning以及时间差学习（Temporal difference learning）

在企业数据应用的场景下， 人们最常用的可能就是监督式学习和非监督式学习的模型。 在图像识别等领域，由于存在大量的非标识的数据和少量的可标识数据， 目前半监督式学习是一个很热的话题。 而强化学习更多的应用在机器人控制及其他需要进行系统控制的领域。

## **算法类似性**

## 

根据算法的功能和形式的类似性，我们可以把算法分类，比如说基于树的算法，基于神经网络的算法等等。当然，机器学习的范围非常庞大，有些算法很难明确归类到某一类。而对于有些分类来说，同一分类的算法可以针对不同类型的问题。这里，我们尽量把常用的算法按照最容易理解的方式进行分类。

### **回归算法：**

![](/assets/mlpre5.png)

回归算法是试图采用对误差的衡量来探索变量之间的关系的一类算法。回归算法是统计机器学习的利器。在机器学习领域，人们说起回归，有时候是指一类问题，有时候是指一类算法，这一点常常会使初学者有所困惑。常见的回归算法包括：最小二乘法（Ordinary Least Square），逻辑回归（Logistic Regression），逐步式回归（Stepwise Regression），多元自适应回归样条（Multivariate Adaptive Regression Splines）以及本地散点平滑估计（Locally Estimated Scatterplot Smoothing）

### **基于实例的算法**

![](/assets/mlpre6.png)

基于实例的算法常常用来对决策问题建立模型，这样的模型常常先选取一批样本数据，然后根据某些近似性把新数据与样本数据进行比较。通过这种方式来寻找最佳的匹配。因此，基于实例的算法常常也被称为“赢家通吃”学习或者“基于记忆的学习”。常见的算法包括 k-Nearest Neighbor\(KNN\), 学习矢量量化（Learning Vector Quantization， LVQ），以及自组织映射算法（Self-Organizing Map ， SOM）

### **正则化方法**

![](/assets/mlpre7.png)

正则化方法是其他算法（通常是回归算法）的延伸，根据算法的复杂度对算法进行调整。正则化方法通常对简单模型予以奖励而对复杂算法予以惩罚。常见的算法包括：Ridge Regression， Least Absolute Shrinkage and Selection Operator（LASSO），以及弹性网络（Elastic Net）。

### **决策树学习**

![](/assets/mlpre8.png)

决策树算法根据数据的属性采用树状结构建立决策模型， 决策树模型常常用来解决分类和回归问题。常见的算法包括：分类及回归树（Classification And Regression Tree， CART）， ID3 \(Iterative Dichotomiser 3\)， C4.5， Chi-squared Automatic Interaction Detection\(CHAID\), Decision Stump, 随机森林（Random Forest）， 多元自适应回归样条（MARS）以及梯度推进机（Gradient Boosting Machine， GBM）

### **贝叶斯方法**

[![](/assets/mlpre9.png "2class\_gauss\_points")](http://www.ctocio.com/hotnews/15919.html/attachment/2class_gauss_points)

贝叶斯方法算法是基于贝叶斯定理的一类算法，主要用来解决分类和回归问题。常见算法包括：朴素贝叶斯算法，平均单依赖估计（Averaged One-Dependence Estimators， AODE），以及Bayesian Belief Network（BBN）。

### **基于核的算法**

![](/assets/mlpre11.png)

基于核的算法中最著名的莫过于支持向量机（SVM）了。 基于核的算法把输入数据映射到一个高阶的向量空间， 在这些高阶向量空间里， 有些分类或者回归问题能够更容易的解决。 常见的基于核的算法包括：支持向量机（Support Vector Machine， SVM）， 径向基函数（Radial Basis Function ，RBF\)， 以及线性判别分析（Linear Discriminate Analysis ，LDA\)等

### **聚类算法**

![](/assets/mlpre12.png)

聚类，就像回归一样，有时候人们描述的是一类问题，有时候描述的是一类算法。聚类算法通常按照中心点或者分层的方式对输入数据进行归并。所以的聚类算法都试图找到数据的内在结构，以便按照最大的共同点将数据进行归类。常见的聚类算法包括 k-Means算法以及期望最大化算法（Expectation Maximization， EM）。

### **关联规则学习**

![](/assets/mlpre13.png)

关联规则学习通过寻找最能够解释数据变量之间关系的规则，来找出大量多元数据集中有用的关联规则。常见算法包括 Apriori算法和Eclat算法等。

### **人工神经网络**

![](/assets/mlpre14.png)

人工神经网络算法模拟生物神经网络，是一类模式匹配算法。通常用于解决分类和回归问题。人工神经网络是机器学习的一个庞大的分支，有几百种不同的算法。（其中[深度学习](http://lib.csdn.net/base/deeplearning)就是其中的一类算法，我们会单独讨论），重要的人工神经网络算法包括：感知器神经网络（Perceptron Neural Network）, 反向传递（Back Propagation）， Hopfield网络，自组织映射（Self-Organizing Map, SOM）。学习矢量量化（Learning Vector Quantization， LVQ）

### **深度学习**

![](/assets/mlpre15.png)

深度学习算法是对人工神经网络的发展。 在近期赢得了很多关注， 特别是[百度也开始发力深度学习后](http://www.ctocio.com/ccnews/15615.html)， 更是在国内引起了很多关注。   在计算能力变得日益廉价的今天，深度学习试图建立大得多也复杂得多的神经网络。很多深度学习的算法是半监督式学习算法，用来处理存在少量未标识数据的[大数据](http://lib.csdn.net/base/hadoop)集。常见的深度学习算法包括：受限波尔兹曼机（Restricted Boltzmann Machine， RBN）， Deep Belief Networks（DBN），卷积网络（Convolutional Network）, 堆栈式自动编码器（Stacked Auto-encoders）。

### **降低维度算法**

![](/assets/mlpre16.png)

像聚类算法一样，降低维度算法试图分析数据的内在结构，不过降低维度算法是以非监督学习的方式试图利用较少的信息来归纳或者解释数据。这类算法可以用于高维数据的可视化或者用来简化数据以便监督式学习使用。常见的算法包括：主成份分析（Principle Component Analysis， PCA），偏最小二乘回归（Partial Least Square Regression，PLS）， Sammon映射，多维尺度（Multi-Dimensional Scaling, MDS）,  投影追踪（Projection Pursuit）等。

### **集成算法：**

![](/assets/mlpre17.png)

集成算法用一些相对较弱的学习模型独立地就同样的样本进行训练，然后把结果整合起来进行整体预测。集成算法的主要难点在于究竟集成哪些独立的较弱的学习模型以及如何把学习结果整合起来。这是一类非常强大的算法，同时也非常流行。常见的算法包括：Boosting， Bootstrapped Aggregation（Bagging）， AdaBoost，堆叠泛化（Stacked Generalization， Blending），梯度推进机（Gradient Boosting Machine, GBM），随机森林（Random Forest）。

