# scikit-learn随机森林调参



在[Bagging与随机森林算法](/ml/integrate/random-forest.md)中，我们对随机森林\(Random Forest, 以下简称RF）的原理做了总结。本文就从实践的角度对RF做一个总结。重点讲述scikit-learn中RF的调参注意事项，以及和GBDT调参的异同点。

# 1. scikit-learn随机森林类库概述

　　　　在scikit-learn中，RF的分类类是RandomForestClassifier，回归类是RandomForestRegressor。当然RF的变种Extra Trees也有， 分类类ExtraTreesClassifier，回归类ExtraTreesRegressor。由于RF和Extra Trees的区别较小，调参方法基本相同，本文只关注于RF的调参。

　　　　和GBDT的调参类似，RF需要调参的参数也包括两部分，第一部分是Bagging框架的参数，第二部分是CART决策树的参数。下面我们就对这些参数做一个介绍。

# 2.  RF框架参数

　　　　首先我们关注于RF的Bagging框架的参数。这里可以和GBDT对比来学习。在[scikit-learn 梯度提升树\(GBDT\)调参小结](/ml/integrate/scikit-gbdt.md)中我们对GBDT的框架参数做了介绍。GBDT的框架参数比较多，重要的有最大迭代器个数，步长和子采样比例，调参起来比较费力。但是RF则比较简单，这是因为bagging框架里的各个弱学习器之间是没有依赖关系的，这减小的调参的难度。换句话说，达到同样的调参效果，RF调参时间要比GBDT少一些。

　　　　下面我来看看RF重要的Bagging框架的参数，由于RandomForestClassifier和RandomForestRegressor参数绝大部分相同，这里会将它们一起讲，不同点会指出。

　　　　1\) **n\_estimators**: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n\_estimators太小，容易欠拟合，n\_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是100。在实际调参的过程中，我们常常将n\_estimators和下面介绍的参数learning\_rate一起考虑。

　　　　2\)**oob\_score**:即是否采用袋外样本来评估模型的好坏。默认识False。个人推荐设置为True，因为袋外分数反应了一个模型拟合后的泛化能力。

　　　　3\) **criterion:**即CART树做划分时对特征的评价标准。分类模型和回归模型的损失函数是不一样的。分类RF对应的CART分类树默认是基尼系数gini,另一个可选择的标准是信息增益。回归RF对应的CART回归树默认是均方差mse，另一个可以选择的标准是绝对值差mae。一般来说选择默认的标准就已经很好的。

　　　　从上面可以看出， RF重要的框架参数比较少，主要需要关注的是 n\_estimators，即RF最大的决策树个数。

# 3.  RF决策树参数

　　　　下面我们再来看RF的决策树参数，它要调参的参数基本和GBDT相同，如下:

　　　　1\) RF划分时考虑的最大特征数**max\_features**: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑$$log_2N$$个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑$$\sqrt{N}$$个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

　　　　2\) 决策树最大深度**max\_depth**: 默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

　　　　3\) 内部节点再划分所需最小样本数**min\_samples\_split**: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min\_samples\_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

　　　　4\) 叶子节点最少样本数**min\_samples\_leaf**: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

　　　　5）叶子节点最小的样本权重和**min\_weight\_fraction\_leaf**：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

　　　　6\) 最大叶子节点数**max\_leaf\_nodes**: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。

　　　　7\) 节点划分最小不纯度**min\_impurity\_split: ** 这个值限制了决策树的增长，如果某节点的不纯度\(基于基尼系数，均方差\)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。一般不推荐改动默认值1e-7。

　　　　上面决策树参数中最重要的包括最大特征数max\_features， 最大深度max\_depth， 内部节点再划分所需最小样本数min\_samples\_split和叶子节点最少样本数min\_samples\_leaf。

# 4.RF调参实例

　　　　这里仍然使用GBDT调参时同样的数据集来做RF调参的实例。本例我们采用袋外分数来评估我们模型的好坏。

　　　　首先，我们载入需要的类库：

```
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics

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

0    19680  
1      320  
Name: Disbursed, dtype: int64

　　　　接着我们选择好样本特征和类别输出。

```
x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']
```

　　　　不管任何参数，都用默认的，我们拟合下数据看看：

```
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)
print rf0.oob_score_
y_predprob = rf0.predict_proba(X)[:,1]
print "AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob)
```

　　　　输出如下，可见袋外分数已经很高，而且AUC分数也很高。相对于GBDT的默认参数输出，RF的默认参数拟合效果对本例要好一些。

0.98005  
AUC Score \(Train\): 0.999833

 　　　　我们首先对n\_estimators进行网格搜索：

```
param_test1 = {'n_estimators':range(10,71,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```

　　　　输出结果如下：

\(\[mean: 0.80681, std: 0.02236, params: {'n\_estimators': 10},  
  mean: 0.81600, std: 0.03275, params: {'n\_estimators': 20},  
  mean: 0.81818, std: 0.03136, params: {'n\_estimators': 30},  
  mean: 0.81838, std: 0.03118, params: {'n\_estimators': 40},  
  mean: 0.82034, std: 0.03001, params: {'n\_estimators': 50},  
  mean: 0.82113, std: 0.02966, params: {'n\_estimators': 60},  
  mean: 0.81992, std: 0.02836, params: {'n\_estimators': 70}\],  
{'n\_estimators': 60},  
0.8211334476626017\)

　　　　这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max\_depth和内部节点再划分所需最小样本数min\_samples\_split进行网格搜索。

```
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, 
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
```

　　　　输出如下：

\(\[mean: 0.79379, std: 0.02347, params: {'min\_samples\_split': 50, 'max\_depth': 3},  
  mean: 0.79339, std: 0.02410, params: {'min\_samples\_split': 70, 'max\_depth': 3},  
  mean: 0.79350, std: 0.02462, params: {'min\_samples\_split': 90, 'max\_depth': 3},  
  mean: 0.79367, std: 0.02493, params: {'min\_samples\_split': 110, 'max\_depth': 3},  
  mean: 0.79387, std: 0.02521, params: {'min\_samples\_split': 130, 'max\_depth': 3},  
  mean: 0.79373, std: 0.02524, params: {'min\_samples\_split': 150, 'max\_depth': 3},  
  mean: 0.79378, std: 0.02532, params: {'min\_samples\_split': 170, 'max\_depth': 3},  
  mean: 0.79349, std: 0.02542, params: {'min\_samples\_split': 190, 'max\_depth': 3},  
  mean: 0.80960, std: 0.02602, params: {'min\_samples\_split': 50, 'max\_depth': 5},  
  mean: 0.80920, std: 0.02629, params: {'min\_samples\_split': 70, 'max\_depth': 5},  
  mean: 0.80888, std: 0.02522, params: {'min\_samples\_split': 90, 'max\_depth': 5},  
  mean: 0.80923, std: 0.02777, params: {'min\_samples\_split': 110, 'max\_depth': 5},  
  mean: 0.80823, std: 0.02634, params: {'min\_samples\_split': 130, 'max\_depth': 5},  
  mean: 0.80801, std: 0.02637, params: {'min\_samples\_split': 150, 'max\_depth': 5},  
  mean: 0.80792, std: 0.02685, params: {'min\_samples\_split': 170, 'max\_depth': 5},  
  mean: 0.80771, std: 0.02587, params: {'min\_samples\_split': 190, 'max\_depth': 5},  
  mean: 0.81688, std: 0.02996, params: {'min\_samples\_split': 50, 'max\_depth': 7},  
  mean: 0.81872, std: 0.02584, params: {'min\_samples\_split': 70, 'max\_depth': 7},  
  mean: 0.81501, std: 0.02857, params: {'min\_samples\_split': 90, 'max\_depth': 7},  
  mean: 0.81476, std: 0.02552, params: {'min\_samples\_split': 110, 'max\_depth': 7},  
  mean: 0.81557, std: 0.02791, params: {'min\_samples\_split': 130, 'max\_depth': 7},  
  mean: 0.81459, std: 0.02905, params: {'min\_samples\_split': 150, 'max\_depth': 7},  
  mean: 0.81601, std: 0.02808, params: {'min\_samples\_split': 170, 'max\_depth': 7},  
  mean: 0.81704, std: 0.02757, params: {'min\_samples\_split': 190, 'max\_depth': 7},  
  mean: 0.82090, std: 0.02665, params: {'min\_samples\_split': 50, 'max\_depth': 9},  
  mean: 0.81908, std: 0.02527, params: {'min\_samples\_split': 70, 'max\_depth': 9},  
  mean: 0.82036, std: 0.02422, params: {'min\_samples\_split': 90, 'max\_depth': 9},  
  mean: 0.81889, std: 0.02927, params: {'min\_samples\_split': 110, 'max\_depth': 9},  
  mean: 0.81991, std: 0.02868, params: {'min\_samples\_split': 130, 'max\_depth': 9},  
  mean: 0.81788, std: 0.02436, params: {'min\_samples\_split': 150, 'max\_depth': 9},  
  mean: 0.81898, std: 0.02588, params: {'min\_samples\_split': 170, 'max\_depth': 9},  
  mean: 0.81746, std: 0.02716, params: {'min\_samples\_split': 190, 'max\_depth': 9},  
  mean: 0.82395, std: 0.02454, params: {'min\_samples\_split': 50, 'max\_depth': 11},  
  mean: 0.82380, std: 0.02258, params: {'min\_samples\_split': 70, 'max\_depth': 11},  
  mean: 0.81953, std: 0.02552, params: {'min\_samples\_split': 90, 'max\_depth': 11},  
  mean: 0.82254, std: 0.02366, params: {'min\_samples\_split': 110, 'max\_depth': 11},  
  mean: 0.81950, std: 0.02768, params: {'min\_samples\_split': 130, 'max\_depth': 11},  
  mean: 0.81887, std: 0.02636, params: {'min\_samples\_split': 150, 'max\_depth': 11},  
  mean: 0.81910, std: 0.02734, params: {'min\_samples\_split': 170, 'max\_depth': 11},  
  mean: 0.81564, std: 0.02622, params: {'min\_samples\_split': 190, 'max\_depth': 11},  
  mean: 0.82291, std: 0.02092, params: {'min\_samples\_split': 50, 'max\_depth': 13},  
  mean: 0.82177, std: 0.02513, params: {'min\_samples\_split': 70, 'max\_depth': 13},  
  mean: 0.82415, std: 0.02480, params: {'min\_samples\_split': 90, 'max\_depth': 13},  
  mean: 0.82420, std: 0.02417, params: {'min\_samples\_split': 110, 'max\_depth': 13},  
  mean: 0.82209, std: 0.02481, params: {'min\_samples\_split': 130, 'max\_depth': 13},  
  mean: 0.81852, std: 0.02227, params: {'min\_samples\_split': 150, 'max\_depth': 13},  
  mean: 0.81955, std: 0.02885, params: {'min\_samples\_split': 170, 'max\_depth': 13},  
  mean: 0.82092, std: 0.02600, params: {'min\_samples\_split': 190, 'max\_depth': 13}\],  
{'max\_depth': 13, 'min\_samples\_split': 110},  
0.8242016800050813\)

　　　　我们看看我们现在模型的袋外分数：

```
rf1 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=110,
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
rf1.fit(X,y)
print rf1.oob_score_
```

　　　　输出结果为：

0.984

　　　　可见此时我们的袋外分数有一定的提高。也就是时候模型的泛化能力增强了。

　　　　对于内部节点再划分所需最小样本数min\_samples\_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需最小样本数min\_samples\_split和叶子节点最少样本数min\_samples\_leaf一起调参。

```
param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, max_depth=13,
                                  max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```

　　　　输出如下：

\(\[mean: 0.82093, std: 0.02287, params: {'min\_samples\_split': 80, 'min\_samples\_leaf': 10},  
  mean: 0.81913, std: 0.02141, params: {'min\_samples\_split': 100, 'min\_samples\_leaf': 10},  
  mean: 0.82048, std: 0.02328, params: {'min\_samples\_split': 120, 'min\_samples\_leaf': 10},  
  mean: 0.81798, std: 0.02099, params: {'min\_samples\_split': 140, 'min\_samples\_leaf': 10},  
  mean: 0.82094, std: 0.02535, params: {'min\_samples\_split': 80, 'min\_samples\_leaf': 20},  
  mean: 0.82097, std: 0.02327, params: {'min\_samples\_split': 100, 'min\_samples\_leaf': 20},  
  mean: 0.82487, std: 0.02110, params: {'min\_samples\_split': 120, 'min\_samples\_leaf': 20},  
  mean: 0.82169, std: 0.02406, params: {'min\_samples\_split': 140, 'min\_samples\_leaf': 20},  
  mean: 0.82352, std: 0.02271, params: {'min\_samples\_split': 80, 'min\_samples\_leaf': 30},  
  mean: 0.82164, std: 0.02381, params: {'min\_samples\_split': 100, 'min\_samples\_leaf': 30},  
  mean: 0.82070, std: 0.02528, params: {'min\_samples\_split': 120, 'min\_samples\_leaf': 30},  
  mean: 0.82141, std: 0.02508, params: {'min\_samples\_split': 140, 'min\_samples\_leaf': 30},  
  mean: 0.82278, std: 0.02294, params: {'min\_samples\_split': 80, 'min\_samples\_leaf': 40},  
  mean: 0.82141, std: 0.02547, params: {'min\_samples\_split': 100, 'min\_samples\_leaf': 40},  
  mean: 0.82043, std: 0.02724, params: {'min\_samples\_split': 120, 'min\_samples\_leaf': 40},  
  mean: 0.82162, std: 0.02348, params: {'min\_samples\_split': 140, 'min\_samples\_leaf': 40},  
  mean: 0.82225, std: 0.02431, params: {'min\_samples\_split': 80, 'min\_samples\_leaf': 50},  
  mean: 0.82225, std: 0.02431, params: {'min\_samples\_split': 100, 'min\_samples\_leaf': 50},  
  mean: 0.81890, std: 0.02458, params: {'min\_samples\_split': 120, 'min\_samples\_leaf': 50},  
  mean: 0.81917, std: 0.02528, params: {'min\_samples\_split': 140, 'min\_samples\_leaf': 50}\],  
{'min\_samples\_leaf': 20, 'min\_samples\_split': 120},  
0.8248650279471544\)

　　　　最后我们再对最大特征数max\_features做调参:

```
param_test4 = {'max_features':range(3,11,2)}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20 ,oob_score=True, random_state=10),
   param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```

　　　　输出如下：

\(\[mean: 0.81981, std: 0.02586, params: {'max\_features': 3},  
  mean: 0.81639, std: 0.02533, params: {'max\_features': 5},  
  mean: 0.82487, std: 0.02110, params: {'max\_features': 7},  
  mean: 0.81704, std: 0.02209, params: {'max\_features': 9}\],  
{'max\_features': 7},  
0.8248650279471544\)

　　　　用我们搜索到的最佳参数，我们再看看最终的模型拟合：

```
rf2 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
rf2.fit(X,y)
print rf2.oob_score_
```

　　　　此时的输出为：

0.984

　　　　可见此时模型的袋外分数基本没有提高，主要原因是0.984已经是一个很高的袋外分数了，如果想进一步需要提高模型的泛化能力，我们需要更多的数据。

