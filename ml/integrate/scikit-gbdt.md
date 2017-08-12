# scikit-learn 梯度提升树\(GBDT\)

---

在[梯度提升树\(GBDT\)](/ml/integrate/gbdt.md)中，我们对GBDT的原理做了总结，本文我们就从scikit-learn里GBDT的类库使用方法作一个总结，主要会关注调参中的一些要点。

# 1. scikit-learn GBDT类库概述

在sacikit-learn中，GradientBoostingClassifier为GBDT的分类类， 而GradientBoostingRegressor为GBDT的回归类。两者的参数类型完全相同，当然有些参数比如损失函数loss的可选择项并不相同。这些参数中，类似于Adaboost，我们把重要参数分为两类，第一类是Boosting框架的重要参数，第二类是弱学习器即CART回归树的重要参数。

下面我们就从这两个方面来介绍这些参数的使用。

# 2. GBDT类库boosting框架参数

首先，我们来看boosting框架相关的重要参数。由于GradientBoostingClassifier和GradientBoostingRegressor的参数绝大部分相同，我们下面会一起来讲，不同点会单独指出。

1\) **n\_estimators**: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n\_estimators太小，容易欠拟合，n\_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是100。在实际调参的过程中，我们常常将n\_estimators和下面介绍的参数learning\_rate一起考虑。

2\) **learning\_rate**: 即每个弱学习器的权重缩减系数ν，也称作步长，在原理篇的正则化章节我们也讲到了，加上了正则化项，我们的强学习器的迭代公式为$$f_{k(x)}=f_{k-1}(x)+vh_{k(x)}$$。ν的取值范围为0&lt;ν≤1。对于同样的训练集拟合效果，较小的ν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n\_estimators和learning\_rate要一起调参。一般来说，可以从一个小一点的ν开始调参，默认是1。

3\)**subsample**: 即我们在原理篇的正则化章节讲到的子采样，取值为\(0,1\]。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在\[0.5, 0.8\]之间，默认是1.0，即不使用子采样。

4\)**init**: 即我们的初始化的时候的弱学习器，拟合对应原理篇里面的$$f_0(x)$$，如果不输入，则用训练集样本来做样本集的初始化分类回归预测。否则用init参数提供的学习器做初始化分类回归预测。一般用在我们对数据有先验知识，或者之前做过一些拟合的时候，如果没有的话就不用管这个参数了。

5\) **loss:**即我们GBDT算法中的损失函数。分类模型和回归模型的损失函数是不一样的。

对于分类模型，有对数似然损失函数"deviance"和指数损失函数"exponential"两者输入选择。默认是对数似然损失函数"deviance"。在原理篇中对这些分类损失函数有详细的介绍。一般来说，推荐使用默认的"deviance"。它对二元分离和多元分类各自都有比较好的优化。而指数损失函数等于把我们带到了Adaboost算法。

对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。默认是均方差"ls"。一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。而如果我们需要对训练集进行分段预测的时候，则采用“quantile”。

6\) **alpha：**这个参数只有GradientBoostingRegressor有，当我们使用Huber损失"huber"和分位数损失“quantile”时，需要指定分位数的值。默认是0.9，如果噪音点较多，可以适当降低这个分位数的值。

# 3. GBDT类库弱学习器参数

这里我们再对GBDT的类库弱学习器的重要参数做一个总结。由于GBDT使用了CART回归决策树，因此它的参数基本来源于决策树类，也就是说，和DecisionTreeClassifier和DecisionTreeRegressor的参数基本类似。如果你已经很熟悉决策树算法的调参，那么这一节基本可以跳过。不熟悉的朋友可以继续看下去。

1\) 划分时考虑的最大特征数max\_features: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑$$log_2N$$个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑$$\sqrt{N}$$个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

2\) 决策树最大深度**max\_depth**: 默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

3\) 内部节点再划分所需最小样本数**min\_samples\_split**: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min\_samples\_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

4\) 叶子节点最少样本数**min\_samples\_leaf**: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

5）叶子节点最小的样本权重和**min\_weight\_fraction\_leaf**：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

6\) 最大叶子节点数**max\_leaf\_nodes**: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。

7\) 节点划分最小不纯度**min\_impurity\_split: ** 这个值限制了决策树的增长，如果某节点的不纯度\(基于基尼系数，均方差\)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。一般不推荐改动默认值1e-7。

# 4. GBDT调参实例

这里我们用一个二元分类的例子来讲解下GBDT的调参。这部分参考了这个Github上的数据调参过程[Parameter\_Tuning\_GBM\_with\_Example](https://github.com/aarshayj/Analytics_Vidhya/tree/master/Articles/Parameter_Tuning_GBM_with_Example)。这个例子的数据有87000多行，单机跑会比较慢，下面的例子我只选择了它的前面20000行，我将其打包后，[下载地址在这](http://files.cnblogs.com/files/pinard/train_modified.zip)。

首先，我们载入需要的类库：

```
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
%matplotlib inline
```

接着，我们把解压的数据用下面的代码载入，顺便看看数据的类别分布。

```
train = pd.read_csv('train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
train['Disbursed'].value_counts() 
```

可以看到类别输出如下，也就是类别0的占大多数。

```
0    19680
1      320
Name: Disbursed, dtype: int64
```

现在我们得到我们的训练集。最后一列Disbursed是分类输出。前面的所有列（不考虑ID列）都是样本特征。

```
x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']
```

不管任何参数，都用默认的，我们拟合下数据看看：

```
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X,y)
y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
```

输出如下，可见拟合还可以，我们下面看看怎么通过调参提高模型的泛化能力。

```
Accuracy : 0.9852
AUC Score (Train): 0.900531
```

首先我们从步长\(learning rate\)和迭代次数\(n\_estimators\)入手。一般来说,开始选择一个较小的步长来网格搜索最好的迭代次数。这里，我们将步长初始值设置为0.1。对于迭代次数进行网格搜索如下：

```
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```

输出如下，可见最好的迭代次数是60。

```
([mean: 0.81285, std: 0.01967, params: {'n_estimators': 20},
  mean: 0.81438, std: 0.01947, params: {'n_estimators': 30},
  mean: 0.81451, std: 0.01933, params: {'n_estimators': 40},
  mean: 0.81618, std: 0.01848, params: {'n_estimators': 50},
  mean: 0.81751, std: 0.01736, params: {'n_estimators': 60},
  mean: 0.81547, std: 0.01900, params: {'n_estimators': 70},
  mean: 0.81299, std: 0.01860, params: {'n_estimators': 80}],
 {'n_estimators': 60},
 0.8175146087398375)
```

找到了一个合适的迭代次数，现在我们开始对决策树进行调参。首先我们对决策树最大深度max\_depth和内部节点再划分所需最小样本数min\_samples\_split进行网格搜索。

```
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20, 
      max_features='sqrt', subsample=0.8, random_state=10), 
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
```

输出如下，可见最好的最大树深度是7，内部节点再划分所需最小样本数是300。

```
([mean: 0.81199, std: 0.02073, params: {'min_samples_split': 100, 'max_depth': 3},
  mean: 0.81267, std: 0.01985, params: {'min_samples_split': 300, 'max_depth': 3},
  mean: 0.81238, std: 0.01937, params: {'min_samples_split': 500, 'max_depth': 3},
  mean: 0.80925, std: 0.02051, params: {'min_samples_split': 700, 'max_depth': 3},
  mean: 0.81846, std: 0.01843, params: {'min_samples_split': 100, 'max_depth': 5},
  mean: 0.81630, std: 0.01810, params: {'min_samples_split': 300, 'max_depth': 5},
  mean: 0.81315, std: 0.01898, params: {'min_samples_split': 500, 'max_depth': 5},
  mean: 0.81262, std: 0.02090, params: {'min_samples_split': 700, 'max_depth': 5},
  mean: 0.81807, std: 0.02004, params: {'min_samples_split': 100, 'max_depth': 7},
  mean: 0.82137, std: 0.01733, params: {'min_samples_split': 300, 'max_depth': 7},
  mean: 0.81703, std: 0.01773, params: {'min_samples_split': 500, 'max_depth': 7},
  mean: 0.81383, std: 0.02327, params: {'min_samples_split': 700, 'max_depth': 7},
  mean: 0.81107, std: 0.02178, params: {'min_samples_split': 100, 'max_depth': 9},
  mean: 0.80944, std: 0.02612, params: {'min_samples_split': 300, 'max_depth': 9},
  mean: 0.81476, std: 0.01973, params: {'min_samples_split': 500, 'max_depth': 9},
  mean: 0.81601, std: 0.02576, params: {'min_samples_split': 700, 'max_depth': 9},
  mean: 0.81091, std: 0.02227, params: {'min_samples_split': 100, 'max_depth': 11},
  mean: 0.81309, std: 0.02696, params: {'min_samples_split': 300, 'max_depth': 11},
  mean: 0.81713, std: 0.02379, params: {'min_samples_split': 500, 'max_depth': 11},
  mean: 0.81347, std: 0.02702, params: {'min_samples_split': 700, 'max_depth': 11},
  mean: 0.81444, std: 0.01813, params: {'min_samples_split': 100, 'max_depth': 13},
  mean: 0.80825, std: 0.02291, params: {'min_samples_split': 300, 'max_depth': 13},
  mean: 0.81923, std: 0.01693, params: {'min_samples_split': 500, 'max_depth': 13},
  mean: 0.81382, std: 0.02258, params: {'min_samples_split': 700, 'max_depth': 13}],
 {'max_depth': 7, 'min_samples_split': 300},
 0.8213724275914632)
```

由于决策树深度7是一个比较合理的值，我们把它定下来，对于内部节点再划分所需最小样本数min\_samples\_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需最小样本数min\_samples\_split和叶子节点最少样本数min\_samples\_leaf一起调参。

```
param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,
                                     max_features='sqrt', subsample=0.8, random_state=10), 
                       param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```

输出结果如下，可见这个min\_samples\_split在边界值，还有进一步调试小于边界60的必要。由于这里只是例子，所以大家可以自己下来用包含小于60的网格搜索来寻找合适的值。

```
([mean: 0.81828, std: 0.02251, params: {'min_samples_split': 800, 'min_samples_leaf': 60},
  mean: 0.81731, std: 0.02344, params: {'min_samples_split': 1000, 'min_samples_leaf': 60},
  mean: 0.82220, std: 0.02250, params: {'min_samples_split': 1200, 'min_samples_leaf': 60},
  mean: 0.81447, std: 0.02125, params: {'min_samples_split': 1400, 'min_samples_leaf': 60},
  mean: 0.81495, std: 0.01626, params: {'min_samples_split': 1600, 'min_samples_leaf': 60},
  mean: 0.81528, std: 0.02140, params: {'min_samples_split': 1800, 'min_samples_leaf': 60},
  mean: 0.81590, std: 0.02517, params: {'min_samples_split': 800, 'min_samples_leaf': 70},
  mean: 0.81573, std: 0.02207, params: {'min_samples_split': 1000, 'min_samples_leaf': 70},
  mean: 0.82021, std: 0.02521, params: {'min_samples_split': 1200, 'min_samples_leaf': 70},
  mean: 0.81512, std: 0.01995, params: {'min_samples_split': 1400, 'min_samples_leaf': 70},
  mean: 0.81395, std: 0.02081, params: {'min_samples_split': 1600, 'min_samples_leaf': 70},
  mean: 0.81587, std: 0.02082, params: {'min_samples_split': 1800, 'min_samples_leaf': 70},
  mean: 0.82064, std: 0.02698, params: {'min_samples_split': 800, 'min_samples_leaf': 80},
  mean: 0.81490, std: 0.02475, params: {'min_samples_split': 1000, 'min_samples_leaf': 80},
  mean: 0.82009, std: 0.02568, params: {'min_samples_split': 1200, 'min_samples_leaf': 80},
  mean: 0.81850, std: 0.02226, params: {'min_samples_split': 1400, 'min_samples_leaf': 80},
  mean: 0.81855, std: 0.02099, params: {'min_samples_split': 1600, 'min_samples_leaf': 80},
  mean: 0.81666, std: 0.02249, params: {'min_samples_split': 1800, 'min_samples_leaf': 80},
  mean: 0.81960, std: 0.02437, params: {'min_samples_split': 800, 'min_samples_leaf': 90},
  mean: 0.81560, std: 0.02235, params: {'min_samples_split': 1000, 'min_samples_leaf': 90},
  mean: 0.81936, std: 0.02542, params: {'min_samples_split': 1200, 'min_samples_leaf': 90},
  mean: 0.81362, std: 0.02254, params: {'min_samples_split': 1400, 'min_samples_leaf': 90},
  mean: 0.81429, std: 0.02417, params: {'min_samples_split': 1600, 'min_samples_leaf': 90},
  mean: 0.81299, std: 0.02262, params: {'min_samples_split': 1800, 'min_samples_leaf': 90},
  mean: 0.82000, std: 0.02511, params: {'min_samples_split': 800, 'min_samples_leaf': 100},
  mean: 0.82209, std: 0.01816, params: {'min_samples_split': 1000, 'min_samples_leaf': 100},
  mean: 0.81821, std: 0.02337, params: {'min_samples_split': 1200, 'min_samples_leaf': 100},
  mean: 0.81922, std: 0.02377, params: {'min_samples_split': 1400, 'min_samples_leaf': 100},
  mean: 0.81545, std: 0.02221, params: {'min_samples_split': 1600, 'min_samples_leaf': 100},
  mean: 0.81704, std: 0.02509, params: {'min_samples_split': 1800, 'min_samples_leaf': 100}],
 {'min_samples_leaf': 60, 'min_samples_split': 1200},
 0.8222032996697154)
```

我们调了这么多参数了，终于可以都放到GBDT类里面去看看效果了。现在我们用新参数拟合数据：

```
gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features='sqrt', subsample=0.8, random_state=10)
gbm1.fit(X,y)
y_pred = gbm1.predict(X)
y_predprob = gbm1.predict_proba(X)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
```

输出如下：

```
Accuracy : 0.984
AUC Score (Train): 0.908099
```

对比我们最开始完全不调参的拟合效果，可见精确度稍有下降，主要原理是我们使用了0.8的子采样，20%的数据没有参与拟合。

现在我们再对最大特征数max\_features进行网格搜索。

```
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, subsample=0.8, random_state=10), 
                       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```

输出如下：

```
([mean: 0.82220, std: 0.02250, params: {'max_features': 7},
  mean: 0.82241, std: 0.02421, params: {'max_features': 9},
  mean: 0.82108, std: 0.02302, params: {'max_features': 11},
  mean: 0.82064, std: 0.01900, params: {'max_features': 13},
  mean: 0.82198, std: 0.01514, params: {'max_features': 15},
  mean: 0.81355, std: 0.02053, params: {'max_features': 17},
  mean: 0.81877, std: 0.01863, params: {'max_features': 19}],
 {'max_features': 9},
 0.822412506351626)
```

现在我们再对子采样的比例进行网格搜索：

```
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features=9, random_state=10), 
                       param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(X,y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
```

输出如下：

```
([mean: 0.81828, std: 0.02392, params: {'subsample': 0.6},
  mean: 0.82344, std: 0.02708, params: {'subsample': 0.7},
  mean: 0.81673, std: 0.02196, params: {'subsample': 0.75},
  mean: 0.82241, std: 0.02421, params: {'subsample': 0.8},
  mean: 0.82285, std: 0.02446, params: {'subsample': 0.85},
  mean: 0.81738, std: 0.02236, params: {'subsample': 0.9}],
 {'subsample': 0.7},
 0.8234378969766262)
```

现在我们基本已经得到我们所有调优的参数结果了。这时我们可以减半步长，最大迭代次数加倍来增加我们模型的泛化能力。再次拟合我们的模型：

```
gbm2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
gbm2.fit(X,y)
y_pred = gbm2.predict(X)
y_predprob = gbm2.predict_proba(X)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
```

输出如下：

```
Accuracy : 0.984
AUC Score (Train): 0.905324
```

可以看到AUC分数比起之前的版本稍有下降，这个原因是我们为了增加模型泛化能力，为防止过拟合而减半步长，最大迭代次数加倍，同时减小了子采样的比例，从而减少了训练集的拟合程度。

下面我们继续将步长缩小5倍，最大迭代次数增加5倍，继续拟合我们的模型：

```
gbm3 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
gbm3.fit(X,y)
y_pred = gbm3.predict(X)
y_predprob = gbm3.predict_proba(X)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
```

输出如下，可见减小步长增加迭代次数可以在保证泛化能力的基础上增加一些拟合程度。

```
Accuracy : 0.984
AUC Score (Train): 0.908581
```

最后我们继续步长缩小一半，最大迭代次数增加2倍，拟合我们的模型：

```
gbm4 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1200,max_depth=7, min_samples_leaf =60, 
               min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
gbm4.fit(X,y)
y_pred = gbm4.predict(X)
y_predprob = gbm4.predict_proba(X)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
```

输出如下，此时由于步长实在太小，导致拟合效果反而变差，也就是说，步长不能设置的过小。

```
Accuracy : 0.984
AUC Score (Train): 0.908232
```



