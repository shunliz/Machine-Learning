# 用scikit-learn学习谱聚类

---

在[谱聚类（spectral clustering）原理总结](/ml/cluster/spectral.md)中，我们对谱聚类的原理做了总结。这里我们就对scikit-learn中谱聚类的使用做一个总结。

# 1. scikit-learn谱聚类概述

　　　　在scikit-learn的类库中，sklearn.cluster.SpectralClustering实现了基于Ncut的谱聚类，没有实现基于RatioCut的切图聚类。同时，对于相似矩阵的建立，也只是实现了基于K邻近法和全连接法的方式，没有基于ϵ-邻近法的相似矩阵。最后一步的聚类方法则提供了两种，K-Means算法和 discretize算法。

　　　　对于SpectralClustering的参数，我们主要需要调参的是相似矩阵建立相关的参数和聚类类别数目，它对聚类的结果有很大的影响。当然其他的一些参数也需要理解，在必要时需要修改默认参数。

# 2. SpectralClustering重要参数与调参注意事项

　　　　下面我们就对SpectralClustering的重要参数做一个介绍，对于调参的注意事项会一起介绍。

　　　　1）**n\_clusters**：代表我们在对谱聚类切图时降维到的维数（原理篇第7节的k1），同时也是最后一步聚类算法聚类到的维数\(原理篇第7节的k2\)。也就是说scikit-learn中的谱聚类对这两个参数统一到了一起。简化了调参的参数个数。虽然这个值是可选的，但是一般还是推荐调参选择最优参数。

　　　　2\)**affinity**: 也就是我们的相似矩阵的建立方式。可以选择的方式有三类，第一类是 'nearest\_neighbors'即K邻近法。第二类是'precomputed'即自定义相似矩阵。选择自定义相似矩阵时，需要自己调用set\_params来自己设置相似矩阵。第三类是全连接法，可以使用各种核函数来定义相似矩阵，还可以自定义核函数。最常用的是内置高斯核函数'rbf'。其他比较流行的核函数有‘linear’即线性核函数, ‘poly’即多项式核函数, ‘sigmoid’即sigmoid核函数。如果选择了这些核函数， 对应的核函数参数在后面有单独的参数需要调。自定义核函数我没有使用过，这里就不多讲了。affinity默认是高斯核'rbf'。一般来说，相似矩阵推荐使用默认的高斯核函数。

　　　　3\) 核函数参数**gamma**: 如果我们在affinity参数使用了多项式核函数 'poly'，高斯核函数‘rbf’, 或者'sigmoid'核函数，那么我们就需要对这个参数进行调参。

　　　　多项式核函数中这个参数对应$$K(x,z)=（γx∙z+r)d$$中的γ。一般需要通过交叉验证选择一组合适的γ,r,d

　　　　高斯核函数中这个参数对应$$K(x,z)=exp(γ||x−z||2)$$中的γ。一般需要通过交叉验证选择合适的γ

　　　　sigmoid核函数中这个参数对应$$K(x,z)=tanh（γx∙z+r)$$中的γ。一般需要通过交叉验证选择一组合适的γ,r

γ默认值为1.0，如果我们affinity使用'nearest\_neighbors'或者是'precomputed'，则这么参数无意义。

　　　　4）核函数参数**degree**：如果我们在affinity参数使用了多项式核函数 'poly'，那么我们就需要对这个参数进行调参。这个参数对应$$K(x,z)=（γx∙z+r)d$$中的d。默认是3。一般需要通过交叉验证选择一组合适的γ,r,d

　　　　5）核函数参数**coef0**: 如果我们在affinity参数使用了多项式核函数 'poly'，或者sigmoid核函数，那么我们就需要对这个参数进行调参。

　　　　多项式核函数中这个参数对应$$K(x,z)=（γx∙z+r)d$$中的r。一般需要通过交叉验证选择一组合适的γ,r,d

　　　　sigmoid核函数中这个参数对应$$K\(x,z\)=tanh（γx∙z+r\)$$中的r。一般需要通过交叉验证选择一组合适的γ,r

　　　　coef0默认为1.

　　　　6）**kernel\_params**：如果affinity参数使用了自定义的核函数，则需要通过这个参数传入核函数的参数。

7 \)**n\_neighbors**: 如果我们affinity参数指定为'nearest\_neighbors'即K邻近法，则我们可以通过这个参数指定KNN算法的K的个数。默认是10.我们需要根据样本的分布对这个参数进行调参。如果我们affinity不使用'nearest\_neighbors'，则无需理会这个参数。

　　　　8）**eigen\_solver**:1在降维计算特征值特征向量的时候，使用的工具。有 None, ‘arpack’, ‘lobpcg’, 和‘amg’4种选择。如果我们的样本数不是特别大，无需理会这个参数，使用''None暴力矩阵特征分解即可,如果样本量太大，则需要使用后面的一些矩阵工具来加速矩阵特征分解。它对算法的聚类效果无影响。

　　　　9）**eigen\_tol**：如果eigen\_solver使用了arpack’，则需要通过eigen\_tol指定矩阵分解停止条件。

　　　　10）**assign\_labels**：即最后的聚类方法的选择，有K-Means算法和 discretize算法两种算法可以选择。一般来说，默认的K-Means算法聚类效果更好。但是由于K-Means算法结果受初始值选择的影响，可能每次都不同，如果我们需要算法结果可以重现，则可以使用discretize。

　　　　11）**n\_init**：即使用K-Means时用不同的初始值组合跑K-Means聚类的次数，这个和K-Means类里面n\_init的意义完全相同，默认是10，一般使用默认值就可以。如果你的n\_clusters值较大，则可以适当增大这个值。

　　　　从上面的介绍可以看出，需要调参的部分除了最后的类别数**n\_clusters**，主要是相似矩阵**affinity**的选择，以及对应的相似矩阵参数。当我选定一个相似矩阵构建方法后，调参的过程就是对应的参数交叉选择的过程。对于K邻近法，需要对**n\_neighbors**进行调参，对于全连接法里面最常用的高斯核函数rbf，则需要对**gamma**进行调参。　　　　　

# 3.SpectralClustering实例

　　　　这里我们用一个例子讲述下SpectralClustering的聚类。我们选择最常用的高斯核来建立相似矩阵，用K-Means来做最后的聚类。

　　　　首先我们生成500个6维的数据集，分为5个簇。由于是6维，这里就不可视化了，代码如下：

```
import numpy as np
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
```

　　　　接着我们看看默认的谱聚类的效果：

```
from sklearn.cluster import SpectralClustering
y_pred = SpectralClustering().fit_predict(X)
from sklearn import metrics
print "Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred) 
```

　　　　输出的Calinski-Harabasz分数为：

```
Calinski-Harabasz Score 14908.9325026　　

```

　　　　由于我们使用的是高斯核，那么我们一般需要对n\_clusters和gamma进行调参。选择合适的参数值。代码如下：

```
for index, gamma in enumerate((0.01,0.1,1,10)):
    for index, k in enumerate((3,4,5,6)):
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
        print "Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(X, y_pred) 
```

　　　　输出如下：

Calinski-Harabasz Score with gamma= 0.01 n\_clusters= 3 score: 1979.77096092  
Calinski-Harabasz Score with gamma= 0.01 n\_clusters= 4 score: 3154.01841219  
Calinski-Harabasz Score with gamma= 0.01 n\_clusters= 5 score: 23410.63895  
Calinski-Harabasz Score with gamma= 0.01 n\_clusters= 6 score: 19303.7340877  
Calinski-Harabasz Score with gamma= 0.1 n\_clusters= 3 score: 1979.77096092  
Calinski-Harabasz Score with gamma= 0.1 n\_clusters= 4 score: 3154.01841219  
Calinski-Harabasz Score with gamma= 0.1 n\_clusters= 5 score: 23410.63895  
Calinski-Harabasz Score with gamma= 0.1 n\_clusters= 6 score: 19427.9618944  
Calinski-Harabasz Score with gamma= 1 n\_clusters= 3 score: 687.787319232  
Calinski-Harabasz Score with gamma= 1 n\_clusters= 4 score: 196.926294549  
Calinski-Harabasz Score with gamma= 1 n\_clusters= 5 score: 23410.63895  
Calinski-Harabasz Score with gamma= 1 n\_clusters= 6 score: 19384.9657724  
Calinski-Harabasz Score with gamma= 10 n\_clusters= 3 score: 43.8197355672  
Calinski-Harabasz Score with gamma= 10 n\_clusters= 4 score: 35.2149370067  
Calinski-Harabasz Score with gamma= 10 n\_clusters= 5 score: 29.1784898767  
Calinski-Harabasz Score with gamma= 10 n\_clusters= 6 score: 47.3799111856

　　　　可见最好的n\_clusters是5，而最好的高斯核参数是1或者0.1.

　　　　我们可以看看不输入可选的n\_clusters的时候，仅仅用最优的gamma为0.1时候的聚类效果，代码如下：

```
y_pred = SpectralClustering(gamma=0.1).fit_predict(X)
print "Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred) 
```

　　　　输出为：

```
Calinski-Harabasz Score 14950.4939717

```

　　　　可见n\_clusters一般还是调参选择比较好。

