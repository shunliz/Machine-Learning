# scikit-learn Adaboost类库

---

在[Adaboost原理](/ml/integrate/adaboost.md)中，我们对Adaboost的算法原理做了一个总结。这里我们就从实用的角度对scikit-learn中Adaboost类库的使用做一个小结，重点对调参的注意事项做一个总结。

# 1. Adaboost类库概述

scikit-learn中Adaboost类库比较直接，就是AdaBoostClassifier和AdaBoostRegressor两个，从名字就可以看出AdaBoostClassifier用于分类，AdaBoostRegressor用于回归。

AdaBoostClassifier使用了两种Adaboost分类算法的实现，SAMME和SAMME.R。而AdaBoostRegressor则使用了我们原理篇里讲到的Adaboost回归算法的实现，即Adaboost.R2。

当我们对Adaboost调参时，主要要对两部分内容进行调参，第一部分是对我们的Adaboost的框架进行调参， 第二部分是对我们选择的弱分类器进行调参。两者相辅相成。下面就对Adaboost的两个类：AdaBoostClassifier和AdaBoostRegressor从这两部分做一个介绍。

# 2. AdaBoostClassifier和AdaBoostRegressor框架参数

我们首先来看看AdaBoostClassifier和AdaBoostRegressor框架参数。两者大部分框架参数相同，下面我们一起讨论这些参数，两个类如果有不同点我们会指出。

1）**base\_estimator：**AdaBoostClassifier和AdaBoostRegressor都有，即我们的弱分类学习器或者弱回归学习器。理论上可以选择任何一个分类或者回归学习器，不过需要支持样本权重。我们常用的一般是CART决策树或者神经网络MLP。默认是决策树，即AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier，而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor。另外有一个要注意的点是，如果我们选择的AdaBoostClassifier算法是SAMME.R，则我们的弱分类学习器还需要支持概率预测，也就是在scikit-learn中弱分类学习器对应的预测方法除了predict还需要有predict\_proba。

2）**algorithm**：这个参数只有AdaBoostClassifier有。主要原因是scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R。两者的主要区别是弱学习器权重的度量，SAMME使用了和我们的原理篇里二元分类Adaboost算法的扩展，即用对样本集分类效果作为弱学习器权重，而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重。由于SAMME.R使用了概率度量的连续值，迭代一般比SAMME快，因此AdaBoostClassifier的默认算法algorithm的值也是SAMME.R。我们一般使用默认的SAMME.R就够了，但是要注意的是使用了SAMME.R， 则弱分类学习器参数base\_estimator必须限制使用支持概率预测的分类器。SAMME算法则没有这个限制。

3）**loss**：这个参数只有AdaBoostRegressor有，Adaboost.R2算法需要用到。有线性‘linear’, 平方‘square’和指数 ‘exponential’三种选择, 默认是线性，一般使用线性就足够了，除非你怀疑这个参数导致拟合程度不好。这个值的意义在原理篇我们也讲到了，它对应了我们对第k个弱分类器的中第i个样本的误差的处理，即：

如果是线性误差，则$$e_{ki}= \frac{|y_i - G_k(x_i)|}{E_k}$$；

如果是平方误差，则$$e_{ki}= \frac{(y_i - G_k(x_i))^2}{E_k^2}$$，

如果是指数误差，则$$e_{ki}= 1 - exp（\frac{-y_i + G_k(x_i))}{E_k}）$$，

$$E_k$$为训练集上的最大误差$$E_k= max|y_i - G_k(x_i)|\;i=1,2...m$$

4\) **n\_estimators**： AdaBoostClassifier和AdaBoostRegressor都有，就是我们的弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n\_estimators太小，容易欠拟合，n\_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是50。在实际调参的过程中，我们常常将n\_estimators和下面介绍的参数learning\_rate一起考虑。

5\) **learning\_rate**:  AdaBoostClassifier和AdaBoostRegressor都有，即每个弱学习器的权重缩减系数$$\nu$$，在原理篇的正则化章节我们也讲到了，加上了正则化项，我们的强学习器的迭代公式为$$f_{k}(x) = f_{k-1}(x) + \nu\alpha_kG_k(x)$$。$$\nu$$的取值范围为$$0 < \nu \leq 1$$。对于同样的训练集拟合效果，较小的$$\nu$$意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n\_estimators和learning\_rate要一起调参。一般来说，可以从一个小一点的$$\nu$$开始调参，默认是1。

# 3. AdaBoostClassifier和AdaBoostRegressor弱学习器参数

这里我们再讨论下AdaBoostClassifier和AdaBoostRegressor弱学习器参数，由于使用不同的弱学习器，则对应的弱学习器参数各不相同。这里我们仅仅讨论默认的决策树弱学习器的参数。即CART分类树DecisionTreeClassifier和CART回归树DecisionTreeRegressor。

DecisionTreeClassifier和DecisionTreeRegressor的参数基本类似，在[scikit-learn决策树算法类库使用](/ml/decisiontree/summary.md)这篇文章中我们对这两个类的参数做了详细的解释。这里我们只拿出调参数时需要尤其注意的最重要几个的参数再拿出来说一遍：

1\) 划分时考虑的最大特征数**max\_features**: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；如果是"log2"意味着划分时最多考虑$$log_2N$$个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑$$\sqrt{N}$$个特征。如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

2\) 决策树最大深**max\_depth**: 默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

3\) 内部节点再划分所需最小样本数**min\_samples\_split**: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min\_samples\_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

4\) 叶子节点最少样本数**min\_samples\_leaf**: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

5）叶子节点最小的样本权重和**min\_weight\_fraction\_leaf**：这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

6\) 最大叶子节点数**max\_leaf\_nodes**: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。

# 4. AdaBoostClassifier实战

这里我们用一个具体的例子来讲解AdaBoostClassifier的使用。

首先我们载入需要的类库：

```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
```

接着我们生成一些随机数据来做二元分类，如果对如何产生随机数据不熟悉，在机器学习算法的随机数据生成中有比较详细的介绍。

```
# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1, y1 = make_gaussian_quantiles(cov=2.0,n_samples=500, n_features=2,n_classes=2, random_state=1)
# 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=400, n_features=2, n_classes=2, random_state=1)
#讲两组数据合成一组数据
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))
```

我们通过可视化看看我们的分类数据，它有两个特征，两个输出类别，用颜色区别。

```
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
```

输出为下图：

![](http://images2015.cnblogs.com/blog/1042406/201612/1042406-20161206153456976-469590725.png)

可以看到数据有些混杂，我们现在用基于决策树的Adaboost来做分类拟合。

```
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=200, learning_rate=0.8)
bdt.fit(X, y)
```

这里我们选择了SAMME算法，最多200个弱分类器，步长0.8，在实际运用中你可能需要通过交叉验证调参而选择最好的参数。拟合完了后，我们用网格图来看看它拟合的区域。

```
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show()
```

输出的图如下：

![](http://images2015.cnblogs.com/blog/1042406/201612/1042406-20161206154123460-1185191410.png)

从图中可以看出，Adaboost的拟合效果还是不错的，现在我们看看拟合分数：

```
print "Score:", bdt.score(X,y)
```

输出为：

```
Score: 0.913333333333
```

也就是说拟合训练集数据的分数还不错。当然分数高并不一定好，因为可能过拟合。

现在我们将最大弱分离器个数从200增加到300。再来看看拟合分数。

```
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.8)
bdt.fit(X, y)
print "Score:", bdt.score(X,y)
```

此时的输出为：

```
Score: 0.962222222222
```

这印证了我们前面讲的，弱分离器个数越多，则拟合程度越好，当然也越容易过拟合。

现在我们降低步长，将步长从上面的0.8减少到0.5，再来看看拟合分数。

```
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.5)
bdt.fit(X, y)
print "Score:", bdt.score(X,y)
```

此时的输出为：

```
Score: 0.894444444444
```

可见在同样的弱分类器的个数情况下，如果减少步长，拟合效果会下降。

最后我们看看当弱分类器个数为700，步长为0.7时候的情况：

```
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=600, learning_rate=0.7)
bdt.fit(X, y)
print "Score:", bdt.score(X,y)
```

此时的输出为：

```
Score: 0.961111111111
```

此时的拟合分数和我们最初的300弱分类器，0.8步长的拟合程度相当。也就是说，在我们这个例子中，如果步长从0.8降到0.7，则弱分类器个数要从300增加到700才能达到类似的拟合效果。

