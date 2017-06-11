# 1. scikit-learn 中KNN相关的类库概述

在scikit-learn 中，与近邻法这一大类相关的类库都在sklearn.neighbors包之中。KNN分类树的类是KNeighborsClassifier，KNN回归树的类是KNeighborsRegressor。除此之外，还有KNN的扩展，即限定半径最近邻分类树的类RadiusNeighborsClassifier和限定半径最近邻回归树的类RadiusNeighborsRegressor， 以及最近质心分类算法NearestCentroid。

在这些算法中，KNN分类和回归的类参数完全一样。限定半径最近邻法分类和回归的类的主要参数也和KNN基本一样。

比较特别是的最近质心分类算法，由于它是直接选择最近质心来分类，所以仅有两个参数，距离度量和特征选择距离阈值，比较简单，因此后面就不再专门讲述最近质心分类算法的参数。

另外几个在sklearn.neighbors包中但不是做分类回归预测的类也值得关注。kneighbors\_graph类返回用KNN时和每个样本最近的K个训练集样本的位置。radius\_neighbors\_graph返回用限定半径最近邻法时和每个样本在限定半径内的训练集样本的位置。NearestNeighbors是个大杂烩，它即可以返回用KNN时和每个样本最近的K个训练集样本的位置，也可以返回用限定半径最近邻法时和每个样本最近的训练集样本的位置，常常用在聚类模型中。

# 2. K近邻法和限定半径最近邻法类库参数小结

本节对K近邻法和限定半径最近邻法类库参数做一个总结。包括KNN分类树的类KNeighborsClassifier，KNN回归树的类KNeighborsRegressor， 限定半径最近邻分类树的类RadiusNeighborsClassifier和限定半径最近邻回归树的类RadiusNeighborsRegressor。这些类的重要参数基本相同，因此我们放到一起讲。

![](/images/cluster/table1.png)

![](/images/cluster/table2.png)

![](/images/cluster/table3.png)

![](/images/cluster/table4.png)

# 3. 使用KNeighborsClassifier做分类的实例

## 3.1 生成随机数据

首先，我们生成我们分类的数据，代码如下：

```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets.samples_generator import make_classification
# X为样本特征，Y为样本类别输出， 共1000个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
X, Y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, n_classes=3)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
plt.show()
```

先看看我们生成的数据图如下。由于是随机生成，如果你也跑这段代码，生成的随机数据分布会不一样。下面是我某次跑出的原始数据图。

![](http://images2015.cnblogs.com/blog/1042406/201611/1042406-20161115161802310-1227649461.png)

接着我们用KNN来拟合模型，我们选择K=15，权重为距离远近。代码如下：

```
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors = 15 , weights='distance')
clf.fit(X, Y)
```

最后，我们可视化一下看看我们预测的效果如何，代码如下：

```
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#确认训练集的边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#生成随机数据来做测试集，然后作预测
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# 画出测试集数据
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 也画出所有的训练集数据
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = 15, weights = 'distance')" )
```

生成的图如下，可以看到大多数数据拟合不错，仅有少量的异常点不在范围内。

![](http://images2015.cnblogs.com/blog/1042406/201611/1042406-20161115162352357-756261898.png)

以上就是使用scikit-learn的KNN相关类库的一个总结，希望可以帮到朋友们。

