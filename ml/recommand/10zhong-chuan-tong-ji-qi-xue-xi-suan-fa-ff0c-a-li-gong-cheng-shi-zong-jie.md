**  
1基于CF的推荐算法**

  


1.1算法简介

  


CF（协同过滤）简单来形容就是利用兴趣相投的原理进行推荐，协同过滤主要分两类，一类是基于物品的协同过滤算法，另一种是基于用户的协同过滤算法，这里主要介绍基于物品的协同过滤算法。

  


给定一批用户，及一批物品，记Vi表示不同用户对物品的评分向量，那么物品i与物品j的相关性为：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vuWia3KibT5eu2sbYUgRJSYKA4J7dkyft5ABoL8BASYgiciaxM0Yao2icv1w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


上述公式是利用余弦公式计算相关系数，相关系数的计算还有：杰卡德相关系数、皮尔逊相关系数等。

  


计算用户u对某一物品的偏好，记用户u对物品i的评分为score\(u,i\)，用户u对物品i的协同过滤得分为rec\(u,j\)。

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vleY3C1jdwTAzficQyJENWawFoibjhGdiaS47kkq4nBibBI5JAoQcJ94jRw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



1.2业务实践

  


以购物篮子为例，业务问题：根据用户的历史购买商品记录，给用户推荐一批商品，协同过滤算法实现方法如下。

  


记buyers表示用户购买商品的向量，记为 其中表示全库用户集合，表示用户对商品的得分，定义如下：

  


* Step1：计算物品之间的相关系数

  


记buyersi表示用户购买商品的向量，记buyersi=\(…,bu,i,…\) u∈U为,其中U表示全库用户集合，bu,i表示用户u对商品i的得分，定义如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vlOAIRexZBtSYDtzS3nBL5QYQsyyJxRXWXQt9K1OyFYdAPkxPyVCwOQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


那么商品i与商品j的相关系数如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vLHSicIJNVMm1m6Szk72GvaBLMn8gGKzZKnxwQuTHkiaXt63tLicxVAQbA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


上述公式是是利用余弦公式计算相关性，含义是商品的用户购买向量夹角越小越相似。此外也可以运用皮尔逊、杰卡德、自定义公式计算相关性，这里不一一列举。

  


* Step2：计算用户对商品的协同过滤得分

  


给定一个用户u，设该用户历史购买商品记录的向量为historyu=\(…,hu,i,…\) ,i∈I其中I表示所有商品的集合：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vuvlXVQXbCicrOvqfQEcIoBQKFllz4wL1xjyNvGaicSBQM8iaO8nrib8eDQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


计算给定一个物品j的协同过滤得分为:  


  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vBoXugib33sXb7CLSskxTfYG1ksJcuH7ibDHagQm2F02S39ySnHuqn0ew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


* Step3：给用户推荐商品

  


通过Step2计算用户对全库商品的协同过滤得分，取得分top 10展示给用户。



**2基于关联规则的推荐算法**

  


2.1算法简介

  


基于关联规则的推荐是根据历史数据统计不同规则出现的关系，形如：X-&gt;Y，表示X事件发生后，Y事件会有一定概率发生，这个概率是通过历史数据统计而来。

  


对于一个规则X-&gt;Y，有两个指标对该规则进行衡量。一个是支持度，表示在所有样本数据中，同时包含X和Y样本的占比。另一个是置信度，表示在所有包含X的样本中，包含Y的样本占比。

  


在关联推荐算法中，最主要的是如何找到最大频繁项，业界主要的做法有两种，分别为Apriori算法和FP树。但在互联网海量的用户特征中，使用这些算法挖掘频繁集计算复杂度非常高，下面我们介绍一种在业务当中简单实用的关联规则算法。



2.2业务实践

  


同样是以购物篮子为例，业务场景是：根据用户历史购物记录，给用户推荐商品。下面我们介绍如何构建简单的关联规则推荐算法。

  


* Step1：数据准备

  


首先收集用户展示购买记录，并且关联用户在展示时刻的特征数据。设总样本数量为n条，数据格式如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v25icLlaofaJRgoOp2flZqR1OExNtrOKvpVVlZAEWkub5PmsmbhWgStw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表1：初始数据  


  


其中用户特征可以是用户历史购买的商品ID，也可以是用户属性特征，例如：年龄、性别、居住地等等。

  


* Step2：特征交叉

  


在上表中，对于同一个样本所有特征两两交叉，生成长度为2的特征规则，合并原来的长度为1的特征规则，得到关联规则数据输入表如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vCV6icwUWa3PIhKhHvrSG5O3OaZKKoQjBwI8QcKLsm15vovcLc6VV06g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表2：rule输入数据  


  


上述表中只用长度为1（原始特征）和2（原始特征两两交叉）的规则作为后面rule的候选集，不做长度为3的规则主要的考虑点是降低规则空间复杂度。

  


* Step3：生成关联规则

  


首先把上表的特征展开，使得一个特征一条记录，如下表：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vY2bts226vuoaWkhkEDrahbLxxpyGG9WfZRtgR7S5ONTfve1UWias2lw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


表3：展开数据

  


计算每个规则的支持度，置信度，提升度。首先作变量声明：

  


* f-&gt;i    表示具备特征f的用户购买商品i的事件

* sf,i     表示规则f-&gt;i的支持度

* cf,i     表示规则f-&gt;i的置信度

  


sf,i 计算方法为：统计表3中中同时满足特征=f，商品=i，用户是否购买=0的记录条数记为notbuyersf,i

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vs47yH44SR6w5pxHC8T7zpqZXrCIHcRQryJSWNvtHSRgdaiaLVsknrug/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


规则选择，规则可以通过以下条件进行过滤。

  


条件1：大于等于某个值，参考值取20-100。

条件2：对所有规则的支持度做降序，取75位数为参考值，sf,i大于等于这个值。

条件3：对所有规则的置信度做降序，取75位数为参考值，cf,i大于等于这个值。

  


* Step4：给用户推荐商品

  


给定一个用户u和一个商品i，通过上述方法生成用户u的特征集合记为F. 我们用该用户特征集合下，所有对i有效特征的均值衡量用户u对该物品的购买可能性p\(u,i\)：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09viaNOLz1d7uMVjSKsdg1MP4UnQbibrXYOBYoPUAczVia6oWc0ZXP9XNQHQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


通过上述公式对全库商品求top 10得分的商品推荐给用户。在实际计算当中，并非会进行全库计算，而是采用特征索引技术进行减少大量冗余计算。



**3基于bayes的推荐算法**

  


3.1原理介绍

  


Bayes\(贝叶斯\)定理是关于随机事件A和B的条件概率相互转化的一则定理，贝叶斯公式如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v3xYLpmNzVLia1lCgf0rA1K2ibYz9iaArpiaCaf0RKwrPQb2Mk72nUTU7WA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


上述公式中，P\(Bi\|u\)表示的含义是在发生了事件u的情况下，发生事件Bi的概率，P\(Bi\)表示事件Bi的发生概率，P\(u\)表示事件u的发生概率。

  


如何利用上述定理进行个性化推荐，下面我们举个业务实践的例子。



3.2业务实践

  


以应用商店中应用推荐为例，业务场景：当用户进入应用商店，根据用户已安装应用列表给用户推荐应用。

  


* Step1：问题分解



给定一个用户u，给该用户推荐应用B，根据贝叶斯公式用户安装概率为：



![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vNt6C3HjedgOtlh1iaCTbz23y27cicRelQtfp2oFC46ibp4dDqGDqSdKbQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

设用户的安装列表为{A1  ,…, An}，把用户u看作是事件{A1  ,…, An}，为了简化问题，假设Ak相互独立，那么：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v9EYJBcjb0bpmytIFxw29P8ekr3gXb4ynDhibCx6KIIsYIeTH57hDHOw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


上述式子可以化为：  


  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vnvX86ibuLvgpmVzBWcdNhSibqwcUrzfGIUp0CKH6aIIkbRM5zr5ialbuQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


在推荐场景中，是对一个用户计算不同应用的得分，然后作降序进行推荐。而对于同一个用户P\(u\)是不变的，所以我们可以用以下公式作为排序依据：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vojFo9sXSWALJa2icibLuG3pMPmRTibvssDMEfLvrzd5Eh9PiczHGSQoTKA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


全库的应用集合记为，所以在贝叶斯推荐模型中主要参数有两个集合，分别为：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vYAtwnZON3h1mGGvHWsDPG3vXTB3vmx3y0KxkGTY3MibyvV75lXEH55g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


* Step2：数据准备

  


首先收集历史的用户在应用商店中应用展示记录，并且关联用户在展示时刻的安装列表，数据格式如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vEKxIYKPJTgZzjVDyEJpPhEcCoC6ZJ1poJpXMTic9slWZ3nvsibibR7MrQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表1：初始数据

  


* Step3：模型参数计算

  


参数集合{P\(B\)\|B∈I}的计算：给定一个应用B. 根据表1，首先中“展示应用=B”的样本数，记为showNumsB然后计算“展示应用=B”且“用户是否安装=1”的样本数记为installNumsB 那么：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v4uTfFHO8XF2dRwkU7qq3ukRGeZibbdXicHt7MzdZKn3amvvibfBwfQABA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


参数集合{P\(Ai\|B\)\|B∈I,Ai∈I}给定一个应用B及Ai 根据表1，首先计算“Ai∈已安装列表”且“展示应用=B”的样本数，记为showNumsAi,B . 然后计算“Ai∈已安装列表”且“展示应用=B”且“用户是否安装=1”的样本数，记为installNumsAi,B 那么：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vzETaO8NTOgyHF1fd0fFnGqJz4vEzibuw2keysa0M3zQELsMHXYIdBBg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


在计算P\(Ai\|B\)可能会遇到样本不足的情况，导致计算出异常值，为了避免这类情况，需要根据经验加个最少安装数限制，这里我们定义为最少安装次数为100，那么：

  


![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

  


其中P\(Ai\)是表示在所有用户中，安装了应用Ai用户的占比。

  


* Step4：给用户推荐应用

  


给定一个用户u，及一批候选推荐应用池，我们通过上述方法计算用户u对候选池中每个应用的得分sortScore\(u,B\)，根据这个值做降序，取top 10的应用推荐给用户。



**4基于KNN的推荐算法**

  


4.1算法简介

  


KNN（K最近邻分类算法）是一种在机器学习中比较简单的算法，它的原理如下：对于一个需要分类的物品A，定义某一种刻画物品之间距离方法，找出该物品最邻近k个有已知类别的物品，这k物品中出现最多的类别即为物品A的类别。如下图：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vHicXicaSjl8woFsUOwwW5yA26leUwcR2icfUNU3MP8ibE9BWvJicjHtbbibA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


在KNN算中，最核心的一点是怎么定义物品之间的距离，这里我们简单列举几种计算物品距离的方法：欧式距离、曼哈顿距离、切比雪夫距离、杰卡德系数、夹角余弦、皮尔逊系数。

  


下面介绍KNN在实际业务中的运用。



4.2业务实践

  


业务场景1：以应用商店为例，在用户下载完一个应用时，触发一个“大家还下载”的推荐，下面介绍如何运用knn算法实现这个场景的推荐：

  


首先定义应用的维度向量，一种简单的方法是离散化所有特征，然后进行one-hot编码，得到所有维度取值0/1的向量V，例如：可以把每个用户当做一个维度，如果第n个用户安装了应用A，那么应用A在第n个维度取值为1，否则为0，运用欧式距离可以得到应用A与应用B的距离公式：![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v7nAfP0ktY4kkE419L2wnxmeRW4HVCOSBFW2mib8WE7ugc7wjFjEQtjw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)。

  


给定一个应用A，通过上述公式取距离最小的4个应用出来，在用户下载完应用A以后给该用户推荐这4个应用。



业务场景2：网络购物中，在“猜你喜欢”场景推荐一批物品给用户，通过用户的历史购物清单，运用杰卡德公式计算用户与用户的相关系数：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vJJmkj1ZYoFadbORMbxsJicxvjYhiaEfWaFAEwdWBRGCPJoSeiaWlvfqGQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09veSpWu7dD08ewoqfuyqPFs6ricYhuNZbAsTricPQnO7D1ZiclDic3zx48RQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)表示购买了物品x的用户集合，那么用户u与用户v的距离定义为：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vR771QZXLMUhnQ0gwjywqRwjy7ZeL0wrVGWbSdUzEXEjlZnicMESPkDQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


给定一个用户u，首先找出这个用户最邻近的k个用户，然后在这k个用户中按照购买的用户数对物品进行降序，去除用户u已经购买的物品，取top 10个物品推荐给用户。

  


**5决策树算法**

  


5.1算法简介

  


决策树一种经典的机器学习分类算法，主要的代表算法有ID3、C4.5、CARD，原理可以简单理解为通过对数据总结、归纳得到一系列分类规则，举个简单的例子：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vEWZyxzQWCr5CicW2OlbLBTpw9EO3bDbwwV1jGncmuiabTJuIFZcTxOAg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


在决策树中，一个叶子节点表示一条决策规则（通常叶子节点的类目等于该节点样本最多的类目），决策树算法的目标是得到准确率高的规则，而决策规则的准确率可以用叶子节点的复杂度来衡量。  




5.2复杂度计算

  


下面列举2种常用复杂度的计算方法, 假设有样本集X，总共的类目有n个，pi表示第i个类目的占比。

  


（1） 信息熵：

  


![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

  


上式中，信息熵的值越高，复杂度越高，样本 的不确定性越大。

  


（2）基尼指数：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vLdGOrf8yPvUrBEVXBwOqLM9kvGLZseKI4DVePFRDxP0cjGVOpX6XBQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


上式中，基尼指数越大，复杂度越高，样本的不确定性也就越大。



5.3裂分指标

  


在决策树的生成过程中，每一个节点的裂分都需要考虑选择哪个属性裂分使得系统的复杂度降低越多。不同算法选用的裂分方法有所不同。

  


（1）ID3：信息增益

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vJJtLhXNPiaFOhANTNg6rIMP4GO13ibjPLUj8yx2s9tmnFSemicB4iaGceg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


 其中H\(x\)表示裂分前系统的复杂度，![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vLDHD60CvHOImKNlJd1zLSuNSickvq6eWlb884kEalrSXadTAt25CwHQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)表示裂分后系统的复杂度。该值越大表示裂分方式使得系统更为有序。

  


（2）C4.5：信息增益率

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vYh4LXZTsNic9dkKKnwYWzaAWGH3QLjwGm45SSuxmhL85JLaqia5cYyXw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vsHgwT19x4mECBCKLBiaegCNg6uBGZNvSwyMfmCg50NqhntdmXHQNLPw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v3ILW6sIXiaxZ8vb5JYKcY4HYFXGv5ZEusnUZv44no1ghkFvYnNVo6nQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)表示A属性的第i个取值占比，其中![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vTPegjRD9C89mjltlllw11EtUlBs2juAnGQK4p2lOKQQlRQYZNRCKfA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)表示的意思是属性A的复杂度，该公式除了考虑系统纯度的增量的同时，也考虑了属性A的复杂度。该值越大表示裂分方式使得系统更为有序。（在ID3算法中，由于选择的是信息增益计算系统纯度增量，往往会选择复杂度高的属性进行裂分，复杂度高的属性取值分段会有很多，导致裂分后某些节点只有少量样本而不具备用于预测的统计意义，C4.5基于这个问题加以改进）。

  


（3）CARD：基尼系数

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v61ibicll2WKGPWIOUEkKMs4psba0ayokZCozRBibUYvPsibfnkr98icnLTQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


CARD算法生成的决策树是一个二叉树，每次裂分只裂分两个节点，Gini\(X\|A\)表示裂分后的复杂度，该值越高样本无序性越大，X1,X2是X的裂分后的两个样本集（裂分方法为遍历所有裂分可能，找出Gini\(X\|A\)最小的那个点）。该值越小表示裂分方式使得系统更为有序。



5.4决策树生成

  


输入：![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vW6cNa4YJMc1q2uibOibIPJecCQw1MoeIInh7iaOK2icSa2SJZvSBnicbiaJg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


裂分指标：选择一种裂分指标（信息增益、信息增益率、Gini系数）。

  


节点裂分终止条件：选择节点最小样本数及最大深度。

  


* Step1：选择一个可裂分的节点Di，循环计算所有属性的裂分指标，选取最优的指标使得系统最为有序那个属性作为裂分点，得到数据集Di+1,Di+2,…

* Step2：所有叶子节点是否都达到了裂分的终止条件，是则执行Step3，否则执行step1。

* Step3：减枝

* Step4：返回决策树T



5.5业务实践

  


业务场景：以应用商店中应用个性化推荐为例。

  


Step1：构造用户画像，收集用户历史下载应用记录、已安装应用记录、用户社会属性（年龄、性别、学历、所在城市）。

Step2：构造应用画像，应用画像包括应用ID，应用类型、应用标签、应用安装量排名、应用CTR等。

Step3：样本收集，收集用户历史曝光下载应用记录（字段：用户ID、应用ID、是否下载），并通关用户ID、应用ID与用户画像、应用画像关联起来得到样本数据，得到样本数据（用户ID，应用ID，用户画像，应用画像，是否下载）。

Step4：构造模型训练样本，定义用户画像与应用画像不同类型特征的交叉规则生成模型特征，运用定义好的交叉规则对所有样本生成模型特征，得到模型训练样本（模型特征，是否下载）。

Step5：模型训练，模型训练样本训练CARD算法，得到预测模型。

Step6：模型使用，给定一个用户和应用，根据上述方法生成用户的用户画像及应用的应用画像，然后运用定义好的交叉特征规则生成模型特征，把模型特征代入模型得到预测值。



**6随机森林算法**

  


6.1算法简介

  


随机森林（RF）是决策树与bagging结合一种分类回归算法，它由多颗决策树构成的一个bagging决策系统。当运用RF进行预测时，首先把需要把样本数据输入到每一棵决策树，每个树得到一个叶子节点，预测的时候，如果是回归问题则统计所有树叶子节点的均值，如果是分类问题则求所有树叶子节点类目出现最多的那个类。

  


RF每棵决策树的构建方式如下：

  


Step1：用M表示数据总特征维度，N表示样本数量，m表示特征抽样维度。

Step2：有放回随机抽取N个样本作为这个树的训练样本。

Step3：对训练样本构建决策树，每次裂分前随机抽取m个特征出来，裂分特征在这m个特征中选择一个最优的裂分特征。

Step4：不作减枝直到不能裂分为止。

  


#### ![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vLbnhQcZlQ9onXDR9Gf1ELOpN1sNCOOicY4phXn2FB610hf64DcSTJeQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

####  

#### 6.2业务实践

  


在实际的业务运用中与决策树类似，在前面介绍的决策树业务实践中可以直接用RF算法替代决策树，构造的方法如上所述，重复地随机抽样样本及抽样特征构造多颗决策树，决策树的棵数需要结合分类精度及模型复杂程度判断。



**7基于矩阵分解的推荐算法**

  


7.1算法介绍

  


在推荐算法中，主要解决的问题是找到用户对物品的偏好得分。矩阵分解算法它的基本思想是认为用户对物品的偏好是外在表现，内在是用户对主题的偏好，而主题对不同物品又有不同的权重，通过用户-&gt;主题-&gt;物品这条链路才形成用户对物品的偏好。

  


矩阵分解的公式：U=PQ

  


其中U表示用户对不同物品的偏好矩阵， P表示用户对不同主题的偏好矩阵， Q表示不同主题对应用的权重。

  


7.2模型求解

  


在实际的业务实践中，往往是已知用户对物品的部分偏好得分，求解用户对未知物品的偏好得分。

  


以应用商店广告场景为例：已知用户在端内的物品曝光点击记录，求解用户对不同广告偏好得分。

  


* Step1：根据样本数据构造矩阵U

  


根据样本数据，用户对曝光物品有点击记为1，没有点击记为0，没有曝光过的物品不赋值（记为-），示例如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vTXtqltfhYkcq6J5ugL7GgT4PcfN5EIJibT3jicCxA3bf7kZfF1GBphzQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


* Step2：求解矩阵P和矩阵Q

  


设矩阵U的大小为N×M，主题数定义为K，那么矩阵的大小是N×K，Q矩阵的大小是K×M，构造损失函数，如下：

  


![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

  


其中ui,j表示矩阵U的第i行第j列元素，pi表示矩阵P的第i行，qj表示矩阵Q的第j列，

通过梯度下降法可以求解矩阵P和矩阵Q。  


  


* Step3：预测用户对没有曝光过物品的偏好得分

  


给定一个用户i，需要预测该用户对物品j的偏好得分，公式为：![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v7XpC3BBSECVa1pibZL9q95ka5KkZ5tWDHbOLiaemTDe250pbia7LfeEog/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)。

  


* Step4：如何给用户推荐物品

  


给定一个用户，通过Step3的公式计算该用户对所有物品的偏好得分，取该用户没有曝光过得分排名前10的物品进行推荐。



**8基于BP的推荐算法**

  


8.1算法介绍

  


BP算法是神经网络的一种算法，BP算法网络是有多层网络构成，信号的传播是由第一层网络开始正向传播到下一层网络。以3层神经网络为例，网络结构示例如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vNxSpQYicPI0nCebU0eyEIIrGnkRMMe36jcZSibFhWia0eJ0n2fVnFYlEQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


以3层神经网络关系如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vmnxBKKWSE7TGyN4zI5ha7hZls0Qprhz6IEHcHONvpLlDIoguM4KlOg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


向量X是模型输入变量向量，wi是Li-1层与Li的连接权重矩阵，bi是Li的偏置向量。函数f是一个激活函数，目前业界常用的激活函数有relu、sigmod、tanh，传统BP神经网络函数一般采用sigmod函数，如果采用该函数，那么：

  


#### ![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vR2Bib4gF4e1I2yFZCYibH02yrgEZAfibP9DuP6gyAAGAhqIVKTowcY2VQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

####  

#### 8.2模型求解

  


以个性化推荐场景中点击率预估为例，上述模型参数有w1, w2, w3, b1, b2, b3，我们通过梯度下降法求解这些参数，首先收集样本，取历史用户推荐的数据及用户对推荐反馈的数据作为样本。变量定义如下：

  


* nums 表示收集样本的数量。

* \(Xi,yi\)表示用户第 个样本的数据，Xi表示样本的特征，yi表示点击情况\(0表示没有点击，1表示点击\)。

* Yi 表示上述模型构造中的输出值，是关于w1, w2, w3, b1, b2, b3的变量。

  


损失函数：常用的的定义有两种，一种是交叉熵，另一种均方差，以均方差为例：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vfGaibSKC1Y3ZnDOsXLtqMjYTaepFFaHsNY9agKOGz9mgoicIIsKuibG8A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


通过上述损失函数，运用梯度下降法求解模型参数w1, w2, w3, b1, b2, b3。



**9基于W2V的推荐算法**

  


9.1算法简介

  


W2V是在2013年由Google开源了一款用于词向量计算的工具，该算法提出的场景主要是解决NLP中词向量化的问题，传统对词向量的方法是one-hot编码，one-hot编码存在主要有两点，第一点维度极高无法直接作为模型的输入变量，第二点是词与词之间没有相关性。W2V的诞生解决了这两个问题，W2V是通过神经网络对词进行低纬度的向量化。

  


W2V有两种模型，一个是CBOW模型，另一个是Skip-gram模型。两个模型都是对词进行向量化，区别在于：CBOW是以一个词为输出目标，以该词邻近的词为输入；Skip-gram是以一个词为输入，以该词邻近的词为输出目标。示例如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vUczsHAdPXag7jNu9ajoTGTUvCicymFIzuPYQvIY7xeqDIFObFbpdHFA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


以CBOW模型为例，该模型的结构图如下：  
  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vBOfnrvxuF1ExBRQO3XaH1sEyFHWKFACEVME8T0vHdMzc1NrzZuXnfg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



各层关系为：

  


* Input层：以一个词为输出目标，以该词邻近的词向量为输入。

* Projection层：把Input层的所有向量叠加求和。

* Output层：首先对语料库中的所有词建立哈夫曼树编码（不使用one-hot编码，one-hot编码太稀疏）。然后为每在每个哈夫曼树节点建立一个逻辑斯蒂分类模型，模型的输入都是Projection的输出。



9.1模型训练

  


模型的参数，模型的参数包括所有词的词向量 和哈夫曼树中每个节点的逻辑斯蒂回归参数![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vVUGkLrmLufeF6a6FurejFTthFtorvR40r8NBYLODgXicGjwV2XdERRQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)。

  


哈夫曼树中的每个节点都是一个逻辑斯蒂回归函数，以输出的词作为叶子节点路径中的每个节点的分类（路径走左分支为1，右分支非0）作为训练目标。例如：上图中输出词假设为“足球”，那么路径如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vBSlPKicVRXPmQtGc4CLgGZ3JPVurGoO0cTAWM1EO53c79IhgC2QL3oA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



损失函数的构造，通过交叉熵的构造，以一个样本例，样本的输入词向量求和为XW，输入词为M，该词对应的哈夫曼树路径为T\(M\)，那么该样本的损失函数如下：

  


![](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

  


把所有样本的按照上述公式计算损失函数，求和后得到模型的损失函数：  


  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v4hPzMfarXiaXgAYahDjw98R51YXInAV1gv2IQRRejn0v8znesia3H3Xw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


通过梯度下降法可以求解所有词的词向量vi。



9.2业务实践

  


场景：网络购物场景中，运用W2V+BP进行个性化推荐。

  


* Step1：对物品进行向量化

  


把每个用户看作一篇文章，用户购买物品按照时间序列排序，物品看作词，带入W2V模型得到物品的向量。  


  


* Step2：样本收集

  


收集客户端中，对用户的物品曝光及购买记录，以用户历史购买的物品列表作为用户画像，以给用户曝光物品后用户是否购买为目标变量。

  


* Step3：构造W2V + BP的模型

  


模型的输入有两个，第一个为用户历史购买物品的向量均值，第二个为曝光物品的向量。模型的输出为用户是否购买曝光的物品，中间用BP网络进行连接。

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vCuD0GzQAH8xVSicy5rgwU5ZicQeHd2JzSmgaw22re8HP9rMIaRGhzs1w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


* Step4：模型训练与使用

  


模型训练：目前业界一般使用TF进行实现，BP网络的节点数及层数需要根据训练情况确定。

  


模型使用：给定一个用户u及一个物品i，把用户u购买物品向量均值及物品i的向量作为模型输入，计算物品i的模型得分。重复该操作，计算出用户u所有候选物品的模型得分，根据物品的模型得分降序推荐给用户。



**10基于LR的推荐算法**

  


10.1原理介绍

  


LR（逻辑斯蒂回归）算法的本质是一个线性回归函数，该算法主要用作二分类的场景，例如点击率预估，算法公式如下：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v4jbyPoKmE0PXHILQTx7OUj7hooZDaYFd79WFzhVJdqpSjHnFThiabcA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


其中x是模型的输入

![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vwicUycI5o4ZwibhJdW5hMMTAaJpNrChFaeOhJOia8LHYYLov6SiaIIFPBQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


* xi表示每个维度的输入。

* w是表示模型输入x的系数向量，w=\( w1, w2, …\), wi表示维度xi的权重。



10.2模型求解

  


我们通过梯度下降法求解我们的模型。以点击率预估为例，首先收集样本。变量定义如下：

  


* nums  表示收集样本的数量 。

* \(Xi,yi\)表示用户第 个样本的数据，Xi表示样本的特征，yi表示点击情况\(0表示没有点击，1表示点击\)。

*  Yi 表示模型的预测值，是关于w,b的变量。

  


定义交叉熵损失函数：

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v7GLPibMz58kb2rxCHqrD5mQIwPO8IssxO1HFpYFylnaq3gH8K7icwYyw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

  


通过梯度下降法求解los\(w,b\)最小时对应的w,b即为所求模型参数。



10.3业务实践

  


LR算法在目前推荐系统业界中，流行的做法是大规模离散化特征（one-hot编码），然后带入LR模型，以广告点击率模型为例，步骤如下：

  


* Step1：构造用户画像

  


按照特征类别构造用户画像，对类别下面的所有特征进行离散化处理，例如：用户历史浏览物品记录，用户社会属性，通过模型给用户打的标签等等。

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vUYGSKQN1fD1u6GfwExeYibWsZzFdh57nun912hVgGvBOIts7q8NQ2bw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表：用户画像  


  


* Step2：构造物品画像

  


构造物品画像，同样也是需要划分物品特征类别，类别下面特征离散化处理，例如：物品ID，物品标签，物品热度等等。

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09v5icDPtZnibQPubqyiaQVgwVwB4iaK5h7352CoXUMN1NeiaT5faMVk82OALA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表：用户画像  


  


* Step3：构造场景画像

  


在实际的业务实践中，往往是一个模型需要用到多个场景，不同场景物品的平均点击率差别很大，为了更好地解决不同场景平均点击率不同的问题，往往需要加上场景特征。场景画像一般只有场景ID，在某些特殊场景（例如：搜索列表）可以加上位置信息。

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vWveDWWiapatiasVfosNRN2wStkSCN4vzxicKqvjwrjZQEWtCN0H5eRXMQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表：场景画像

  


* Step4：收集样本数据

  


收集历史曝光点击数据，收集的数据维度包括:用户ID，物品ID，场景ID，是否点击。然后关联用户画像和物品画像得到模型的训练样本数据。

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vOCGwLkCdkDiaZnXFX6tR696pNSfL7BH1uqmvoTIUPwV7dmB8MPT1cGA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表：样本数据

  


* Step5：构造模型特征

  


通过对样本数据构造模型特征得到模型的输入，模型特征分两类，一类是交叉特征，另一类是原始特征。

  


交叉特征：选择用户的类别特征、选择物品的类别特征、场景ID做三个维度的交叉，例如：用户历史点击记录为item1,item2 ， 物品的ID特征为I1，场景特征为scene1，那么生成的交叉特征为item1&I1&scene1,item2&I1&scene1。

  


原始特征：原始特征是指直接把画像特征作为模型的输入特征，一般是把物品的泛化特征作为原始特征，用于物品冷启动特征或场景冷启动特征，例如：物品的CTR、物品的热度、物品的标签等等。

  


![](https://mmbiz.qpic.cn/mmbiz_png/LwZPmXjm4WyhxSw9kgpl7uQVnQFTu09vQ7Sz0ePuWNVZicFBnPVq94Vx6GBTJTEedrBntVxeiagKqcF1icfvPw82Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表：模型输入

  


* Step6：模型训练

  


把模型中的所有特征进行one-hot编码，假设模型特征数为N，首先给每个模型特征一个唯一1-N的编码，那么每个样本的模型输入向量是维度为N取值0/1的向量 ，0表示该样本具备对应编号的特征，1表示没有，例如：样本1的具有有编号为1和编号为3的特征，那么样本1的模型输入向量为\(1,0,1,0,0,…\)，然后通过通用的LR训练器训练模型，即可把模型的参数训练出来。

  


* Step7：模型使用

  


给定一个用户u，及一批候选物品，对用户u如何推荐物品。通过上述方法计算用户u对候选集中每个物品的模型得分，按照模型得分降序推荐给用户。

