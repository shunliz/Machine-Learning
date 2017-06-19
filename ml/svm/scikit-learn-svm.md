## scikit-learn 支持向量机算法库使用小结

---

之前通过一个系列对支持向量机\(以下简称SVM\)算法的原理做了一个总结，本文从实践的角度对scikit-learn SVM算法库的使用做一个小结。scikit-learn SVM算法库封装了libsvm 和 liblinear 的实现，仅仅重写了算法了接口部分。

# 1. scikit-learn SVM算法库使用概述

scikit-learn中SVM的算法库分为两类，一类是分类的算法库，包括SVC， NuSVC，和LinearSVC 3个类。另一类是回归算法库，包括SVR， NuSVR，和LinearSVR 3个类。相关的类都包裹在sklearn.svm模块之中。

对于SVC， NuSVC，和LinearSVC 3个分类的类，SVC和 NuSVC差不多，区别仅仅在于对损失的度量方式不同，而LinearSVC从名字就可以看出，他是线性分类，也就是不支持各种低维到高维的核函数，仅仅支持线性核函数，对线性不可分的数据不能使用。

同样的，对于SVR， NuSVR，和LinearSVR 3个回归的类， SVR和NuSVR差不多，区别也仅仅在于对损失的度量方式不同。LinearSVR是线性回归，只能使用线性核函数。

我们使用这些类的时候，如果有经验知道数据是线性可以拟合的，那么使用LinearSVC去分类 或者LinearSVR去回归，它们不需要我们去慢慢的调参去选择各种核函数以及对应参数， 速度也快。如果我们对数据分布没有什么经验，一般使用SVC去分类或者SVR去回归，这就需要我们选择核函数以及对核函数调参了。

什么特殊场景需要使用NuSVC分类 和 NuSVR 回归呢？如果我们对训练集训练的错误率或者说支持向量的百分比有要求的时候，可以选择NuSVC分类 和 NuSVR 。它们有一个参数来控制这个百分比。

这些类的详细使用方法我们在下面再详细讲述。

# 2. 回顾SVM分类算法和回归算法

我们先简要回顾下SVM分类算法和回归算法，因为这里面有些参数对应于算法库的参数，如果不先复习下，下面对参数的讲述可能会有些难以理解。

对于SVM分类算法，其原始形式是：


$$
min\;\; \frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i
$$



$$
s.t. \;\; y_i(w \bullet \phi(x_i) + b) \geq 1 - \xi_i \;\;(i =1,2,...m)
$$



$$
\xi_i \geq 0 \;\;(i =1,2,...m)
$$


其中m为样本个数，我们的样本为$$(x_1,y_1),(x_2,y_2),...,(x_m,y_m)$$。w,b是我们的分离超平面的$$w \bullet \phi(x_i) + b = 0$$系数,$$\xi_i$$为第i个样本的松弛系数， C为惩罚系数。$$\phi(x_i)$$为低维到高维的映射函数。

通过拉格朗日函数以及对偶化后的形式为：


$$
\underbrace{ min }_{\alpha} \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jK(x_i,x_j) - \sum\limits_{i=1}^{m}\alpha_i
$$



$$
s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0
$$



$$
0 \leq \alpha_i \leq C
$$


其中和原始形式不同的$$\alpha$$为拉格朗日系数向量。$$K(x_i,x_j)$$为我们要使用的核函数。

对于SVM回归算法，其原始形式是：


$$
min\;\; \frac{1}{2}||w||_2^2 + C\sum\limits_{i=1}^{m}(\xi_i^{\lor}+ \xi_i^{\land})
$$



$$
s.t. \;\;\; -\epsilon - \xi_i^{\lor} \leq y_i - w \bullet \phi(x_i ) -b \leq \epsilon + \xi_i^{\land}
$$



$$
\xi_i^{\lor} \geq 0, \;\; \xi_i^{\land} \geq 0 \;(i = 1,2,..., m)
$$


其中m为样本个数，我们的样本为$$(x_1,y_1),(x_2,y_2),...,(x_m,y_m)$$。w,b是我们的回归超平面的$$w \bullet x_i + b = 0$$系数,$$\xi_i^{\lor}， \xi_i^{\land}$$为第i个样本的松弛系数， C为惩罚系数，$$\epsilon$$为损失边界，到超平面距离小于$$\epsilon$$的训练集的点没有损失。$$\phi(x_i)$$为低维到高维的映射函数。

通过拉格朗日函数以及对偶化后的形式为：


$$
\underbrace{ min}_{\alpha^{\lor}, \alpha^{\land}}\; \frac{1}{2}\sum\limits_{i=1,j=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor})(\alpha_j^{\land} - \alpha_j^{\lor})K(x_i,x_j) - \sum\limits_{i=1}^{m}(\epsilon-y_i)\alpha_i^{\land}+ (\epsilon+y_i)\alpha_i^{\lor}
$$



$$
s.t. \; \sum\limits_{i=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor}) = 0
$$



$$
0 < \alpha_i^{\lor} < C \; (i =1,2,...m)
$$



$$
0 < \alpha_i^{\land} < C \; (i =1,2,...m)
$$


其中和原始形式不同的$$\alpha^{\lor}， \alpha^{\land}$$为拉格朗日系数向量。$$K(x_i,x_j)$$为我们要使用的核函数。

# 3. SVM核函数概述

在scikit-learn中，内置的核函数一共有4种，当然如果你认为线性核函数不算核函数的话，那就只有三种。

1）线性核函数（Linear Kernel）表达式为：$$K(x, z) = x \bullet z$$，就是普通的内积，LinearSVC 和 LinearSVR 只能使用它。

2\)  多项式核函数（Polynomial Kernel）是线性不可分SVM常用的核函数之一，表达式为：$$K(x, z) = （\gamma x \bullet z + r)^d$$，其中，$$\gamma$$, r, d都需要自己调参定义,比较麻烦。

3）高斯核函数（Gaussian Kernel），在SVM中也称为径向基核函数（Radial Basis Function,RBF），它是libsvm默认的核函数，当然也是scikit-learn默认的核函数。表达式为：$$K(x, z) = exp(\gamma||x-z||^2)$$， 其中，$$\gamma$$大于0，需要自己调参定义。

4）Sigmoid核函数（Sigmoid Kernel）也是线性不可分SVM常用的核函数之一，表达式为：$$K(x, z) = tanh（\gamma x \bullet z + r)$$， 其中，$$\gamma$$, r都需要自己调参定义。

一般情况下，对非线性数据使用默认的高斯核函数会有比较好的效果，如果你不是SVM调参高手的话，建议使用高斯核来做数据分析。

# 4. SVM分类算法库参数小结

这里我们对SVM分类算法库的重要参数做一个详细的解释，重点讲述调参的一些注意点。

| **参数** | **LinearSVC ** | **SVC** | **NuSVC** |
| :--- | :--- | :--- | :--- |
|  | 惩罚系数C | 即为我们第二节中SVM分类模型原型形式和对偶形式中的惩罚系数C，默认为1，一般需要通过交叉验证来选择一个合适的C。一般来说，如果噪音点较多时，C需要小一些。 | NuSVC没有这个参数, 它通过另一个参数nu来控制训练集训练的错误率，等价于选择了一个C，让训练集训练后满足一个确定的错误率 |
|  | nu | LinearSVC 和SVC没有这个参数，LinearSVC 和SVC使用惩罚系数C来控制惩罚力度。 | nu代表训练集训练的错误率的上限，或者说支持向量的百分比下限，取值范围为\(0,1\],默认是0.5.它和惩罚系数C类似，都可以控制惩罚的力度。 |
|  | 核函数 kernel | LinearSVC没有这个参数，LinearSVC限制了只能使用线性核函数 | 核函数有四种内置选择，第三节已经讲到：‘linear’即线性核函数, ‘poly’即多项式核函数, ‘rbf’即高斯核函数, ‘sigmoid’即sigmoid核函数。如果选择了这些核函数， 对应的核函数参数在后面有单独的参数需要调。默认是高斯核'rbf'。还有一种选择为"precomputed",即我们预先计算出所有的训练集和测试集的样本对应的Gram矩阵，这样K\(x,z\)直接在对应的Gram矩阵中找对应的位置的值。当然我们也可以自定义核函数， 由于我没有用过自定义核函数，这里就不多讲了。 |
|  | 正则化参数penalty | 仅仅对线性拟合有意义，可以选择‘l1’即L1正则化 或者 ‘l2’即L2正则化。默认是L2正则化，如果我们需要产生稀疏话的系数的时候，可以选L1正则化,这和线性回归里面的Lasso回归类似。 | SVC和NuSVC没有这个参数 |
|  | 是否用对偶形式优化dual | 这是一个布尔变量，控制是否使用对偶形式来优化算法，默认是True,即采用上面第二节的分类算法对偶形式来优化算法。如果我们的样本量比特征数多，此时采用对偶形式计算量较大，推荐dual设置为False，即采用原始形式优化 | SVC和NuSVC没有这个参数 |
|  | 核函数参数degree | LinearSVC没有这个参数，LinearSVC限制了只能使用线性核函数 | 如果我们在kernel参数使用了多项式核函数 'poly'，那么我们就需要对这个参数进行调参。这个参数对应$$K(x, z) = （\gamma x \bullet z  + r)^d$$中的d。默认是3。一般需要通过交叉验证选择一组合适的$$\gamma$$, r, d |
|  | 核函数参数gamma | LinearSVC没有这个参数，LinearSVC限制了只能使用线性核函数 | 如果我们在kernel参数使用了多项式核函数 'poly'，高斯核函数‘rbf’, 或者sigmoid核函数，那么我们就需要对这个参数进行调参。多项式核函数中这个参数对应$$K(x, z) = （\gamma x \bullet z  + r)^d$$中的$$\gamma$$。一般需要通过交叉验证选择一组合适的$$\gamma$$, r, d高斯核函数中这个参数对应$$K\(x, z\) = exp\(\gamma\|\|x-z\|\|^2\)$$中的$$\gamma$$。一般需要通过交叉验证选择合适的\gammasigmoid核函数中这个参数对应K\(x, z\) = tanh（\gamma x \bullet z  + r\)中的\gamma。一般需要通过交叉验证选择一组合适的\gamma, r\gamma默认为'auto',即\frac{1}{特征维度} |
|  | 核函数参数coef0 | LinearSVC没有这个参数，LinearSVC限制了只能使用线性核函数 | 如果我们在kernel参数使用了多项式核函数 'poly'，或者sigmoid核函数，那么我们就需要对这个参数进行调参。多项式核函数中这个参数对应K\(x, z\) = （\gamma x \bullet z  + r\)^d中的r。一般需要通过交叉验证选择一组合适的\gamma, r, dsigmoid核函数中这个参数对应K\(x, z\) = tanh（\gamma x \bullet z  + r\)中的r。一般需要通过交叉验证选择一组合适的\gamma, rcoef0默认为0 |
|  |  | 样本权重class\_weight | 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策过于偏向这些类别。这里可以自己指定各个样本的权重，或者用“balanced”，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的"None" |
|  | 分类决策decision\_function\_shape | LinearSVC没有这个参数，使用multi\_class参数替代。 | 可以选择'ovo'或者‘ovo’.目前0.18版本默认是'ovo'.0.19版本将是'ovr'OvR\(one ve rest\)的思想很简单，无论你是多少元分类，我们都可以看做二元分类。具体做法是，对于第K类的分类决策，我们把所有第K类的样本作为正例，除了第K类样本以外的所有样本都作为负例，然后在上面做二元分类，得到第K类的分类模型。其他类的分类模型获得以此类推。OvO\(one-vs-one\)则是每次每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，进行二元分类，得到模型参数。我们一共需要T\(T-1\)/2次分类。从上面的描述可以看出OvR相对简单，但分类效果相对略差（这里指大多数样本分布情况，某些样本分布下OvR可能更好）。而OvO分类相对精确，但是分类速度没有OvR快。一般建议使用OvO以达到较好的分类效果。 |
|  | 分类决策multi\_class | 可以选择 ‘ovr’ 或者 ‘crammer\_singer’ ‘ovr’和SVC和nuSVC中的decision\_function\_shape对应的‘ovr’类似。'crammer\_singer'是一种改良版的'ovr'，说是改良，但是没有比’ovr‘好，一般在应用中都不建议使用。 | SVC和nuSVC没有这个参数，使用decision\_function\_shape参数替代。 |
|  | 缓存大小cache\_size | LinearSVC计算量不大，因此不需要这个参数 | 在大样本的时候，缓存大小会影响训练速度，因此如果机器内存大，推荐用500MB甚至1000MB。默认是200，即200MB. |

# 5. SVM回归算法库参数小结

SVM回归算法库的重要参数巨大部分和分类算法库类似，因此这里重点讲述和分类算法库不同的部分，对于相同的部分可以参考上一节对应参数。

| **参数** | **LinearSVR** | **SVR** | **nuSVR** |
| :--- | :--- | :--- | :--- |
|  |  | 惩罚系数C | 即为我们第二节中SVM分类模型原型形式和对偶形式中的惩罚系数C，默认为1，一般需要通过交叉验证来选择一个合适的C。一般来说，如果噪音点较多时，C需要小一些。大家可能注意到在分类模型里面，nuSVC使用了nu这个等价的参数控制错误率，就没有使用C，为什么我们nuSVR仍然有这个参数呢，不是重复了吗？这里的原因在回归模型里面，我们除了惩罚系数C还有还有一个距离误差\epsilon来控制损失度量，因此仅仅一个nu不能等同于C.也就是说分类错误率是惩罚系数C和距离误差\epsilon共同作用的结果。后面我们可以看到nuSVR中nu的作用。 |
|  | nu | LinearSVR 和SVR没有这个参数，用\epsilon控制错误率 | nu代表训练集训练的错误率的上限，或者说支持向量的百分比下限，取值范围为\(0,1\],默认是0.5.通过选择不同的错误率可以得到不同的距离误差\epsilon。也就是说这里的nu的使用和LinearSVR 和SVR的\epsilon参数等价。 |
|  | 距离误差epsilon | 即我们第二节回归模型中的\epsilon，训练集中的样本需满足-\epsilon - \xi\_i^{\lor} \leq y\_i - w \bullet \phi\(x\_i \) -b \leq \epsilon + \xi\_i^{\land} | nuSVR没有这个参数，用nu控制错误率 |
|  | 是否用对偶形式优化dual | 和SVC类似，可参考上一节的dual描述 | SVR和NuSVR没有这个参数 |
|  | 正则化参数penalty | 和SVC类似，可参考上一节的penalty 描述 | SVR和NuSVR没有这个参数 |
|  | 核函数 kernel | LinearSVR没有这个参数，LinearSVR限制了只能使用线性核函数 | 和SVC, nuSVC类似，可参考上一节的kernel描述 |
|  | 核函数参数degree, gamma 和coef0 | LinearSVR没有这些参数，LinearSVR限制了只能使用线性核函数 | 和SVC, nuSVC类似，可参考上一节的kernel参数描述 |
|  |  | 样本权重class\_weight | 和LinearSVC， SVC, nuSVC类似，可参考上一节的class\_weight描述 |
|  | 损失函数度量loss | 可以选择为‘epsilon\_insensitive’ 和 ‘squared\_epsilon\_insensitive’ ，如果选择‘epsilon\_insensitive’ ，则损失度量满足-\epsilon - \xi\_i^{\lor} \leq y\_i - w \bullet \phi\(x\_i \) -b \leq \epsilon + \xi\_i^{\land}，即和第二节的损失度量一样。是默认的SVM回归的损失度量标准形式。如果选择为 ‘squared\_epsilon\_insensitive’ , 则损失度量满足\(y\_i - w \bullet \phi\(x\_i \) -b\)^2 \leq \epsilon + \xi\_i，此时可见会少一个松弛系数。其优化过程我们在SVM原理系列里没有讲，但是目标函数优化过程是完全相似的。一般用默认的‘epsilon\_insensitive’就足够了。 | SVR和NuSVR没有这个参数 |
|  | 缓存大小cache\_size | LinearSVC计算量不大，因此不需要这个参数 | 在大样本的时候，缓存大小会影响训练速度，因此如果机器内存大，和SVC，nuSVC一样，推荐用500MB甚至1000MB。默认是200，即200MB. |

# 6. SVM算法库其他调参要点

上面已经对scikit-learn中类库的参数做了总结，这里对其他的调参要点做一个小结。

1）一般推荐在做训练之前对数据进行归一化，当然测试集中的数据也需要归一化。。

2）在特征数非常多的情况下，或者样本数远小于特征数的时候，使用线性核，效果已经很好，并且只需要选择惩罚系数C即可。

3）在选择核函数时，如果线性拟合不好，一般推荐使用默认的高斯核'rbf'。这时我们主要需要对惩罚系数C和核函数参数\gamma进行艰苦的调参，通过多轮的交叉验证选择合适的惩罚系数C和核函数参数\gamma。

4）理论上高斯核不会比线性核差，但是这个理论却建立在要花费更多的时间来调参上。所以实际上能用线性核解决问题我们尽量使用线性核。

