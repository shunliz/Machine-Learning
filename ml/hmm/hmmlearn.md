# 用hmmlearn学习隐马尔科夫模型HMM

---

# 1. hmmlearn概述

hmmlearn安装很简单，"pip install hmmlearn"即可完成。

hmmlearn实现了三种HMM模型类，按照观测状态是连续状态还是离散状态，可以分为两类。GaussianHMM和GMMHMM是连续观测状态的HMM模型，而MultinomialHMM是离散观测状态的模型，也是我们在HMM原理系列篇里面使用的模型。

对于MultinomialHMM的模型，使用比较简单，"startprob\_"参数对应我们的隐藏状态初始分布$$\Pi$$, "transmat\_"对应我们的状态转移矩阵A, "emissionprob\_"对应我们的观测状态概率矩阵B。

对于连续观测状态的HMM模型，GaussianHMM类假设观测状态符合高斯分布，而GMMHMM类则假设观测状态符合混合高斯分布。一般情况下我们使用GaussianHMM即高斯分布的观测状态即可。以下对于连续观测状态的HMM模型，我们只讨论GaussianHMM类。

在GaussianHMM类中，"startprob\_"参数对应我们的隐藏状态初始分布$$\Pi$$, "transmat\_"对应我们的状态转移矩阵A, 比较特殊的是观测状态概率的表示方法，此时由于观测状态是连续值，我们无法像MultinomialHMM一样直接给出矩阵B。而是采用给出各个隐藏状态对应的观测状态高斯分布的概率密度函数的参数。

如果观测序列是一维的，则观测状态的概率密度函数是一维的普通高斯分布。如果观测序列是N维的，则隐藏状态对应的观测状态的概率密度函数是N维高斯分布。高斯分布的概率密度函数参数可以用\mu表示高斯分布的期望向量，\Sigma表示高斯分布的协方差矩阵。在GaussianHMM类中，“means”用来表示各个隐藏状态对应的高斯分布期望向量\mu形成的矩阵，而“covars”用来表示各个隐藏状态对应的高斯分布协方差矩阵\Sigma形成的三维张量。

# 2. MultinomialHMM实例

下面我们用我们在[HMM系列](http://www.cnblogs.com/pinard/p/6945257.html)原理篇中的例子来使用MultinomialHMM跑一遍。

首先建立HMM的模型：

[![](http://common.cnblogs.com/images/copycode.gif "复制代码")](javascript:void%280%29;)

```
import
 numpy as np

from
 hmmlearn 
import
 hmm

states 
= [
"
box 1
"
, 
"
box 2
"
, 
"
box3
"
]
n_states 
=
 len(states)

observations 
= [
"
red
"
, 
"
white
"
]
n_observations 
=
 len(observations)

start_probability 
= np.array([0.2, 0.4, 0.4
])

transition_probability 
=
 np.array([
  [
0.5, 0.2, 0.3
],
  [
0.3, 0.5, 0.2
],
  [
0.2, 0.3, 0.5
]
])

emission_probability 
=
 np.array([
  [
0.5, 0.5
],
  [
0.4, 0.6
],
  [
0.7, 0.3
]
])

model 
= hmm.MultinomialHMM(n_components=
n_states)
model.startprob_
=
start_probability
model.transmat_
=
transition_probability
model.emissionprob_
=emission_probability
```

[![](http://common.cnblogs.com/images/copycode.gif "复制代码")](javascript:void%280%29;)

现在我们来跑一跑HMM问题三维特比算法的解码过程，使用和原理篇一样的观测序列来解码，代码如下：

```
seen = np.array([[0,1
,0]]).T
logprob, box 
= model.decode(seen, algorithm=
"
viterbi
"
)

print
(
"
The ball picked:
"
, 
"
, 
"
.join(map(
lambda
 x: observations[x], seen)))

print
(
"
The hidden box
"
, 
"
, 
"
.join(map(
lambda
 x: states[x], box)))
```

输出结果如下：

```
('The ball picked:', 'red, white, red')
('The hidden box', 'box3, box3, box3')
```

可以看出，结果和我们原理篇中的手动计算的结果是一样的。

也可以使用predict函数，结果也是一样的，代码如下：

```
box2 =
 model.predict(seen)

print
(
"
The ball picked:
"
, 
"
, 
"
.join(map(
lambda
 x: observations[x], seen)))

print
(
"
The hidden box
"
, 
"
, 
"
.join(map(
lambda
 x: states[x], box2)))
```

大家可以跑一下，看看结果是否和decode函数相同。

现在我们再来看看求HMM问题一的观测序列的概率的问题，代码如下：

```
print
 model.score(seen)
```

输出结果是：

```
-2.03854530992
```

要注意的是score函数返回的是以自然对数为底的对数概率值，我们在HMM问题一中手动计算的结果是未取对数的原始概率是0.13022。对比一下：ln0.13022 \approx -2.0385

现在我们再看看HMM问题二，求解模型参数的问题。由于鲍姆-韦尔奇算法是基于EM算法的近似算法，所以我们需要多跑几次，比如下面我们跑三次，选择一个比较优的模型参数，代码如下：

[![](http://common.cnblogs.com/images/copycode.gif "复制代码")](javascript:void%280%29;)

```
import
 numpy as np

from
 hmmlearn 
import
 hmm

states 
= [
"
box 1
"
, 
"
box 2
"
, 
"
box3
"
]
n_states 
=
 len(states)

observations 
= [
"
red
"
, 
"
white
"
]
n_observations 
=
 len(observations)
model2 
= hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01
)
X2 
= np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1
]])
model2.fit(X2)

print
 model2.startprob_

print
 model2.transmat_

print
 model2.emissionprob_

print
 model2.score(X2)
model2.fit(X2)

print
 model2.startprob_

print
 model2.transmat_

print
 model2.emissionprob_

print
 model2.score(X2)
model2.fit(X2)

print
 model2.startprob_

print
 model2.transmat_

print
 model2.emissionprob_

print
 model2.score(X2)
```

[![](http://common.cnblogs.com/images/copycode.gif "复制代码")](javascript:void%280%29;)

结果这里就略去了，最终我们会选择分数最高的模型参数。

以上就是用MultinomialHMM解决HMM模型三个问题的方法。

# 3. GaussianHMM实例

下面我们再给一个GaussianHMM的实例，这个实例中，我们的观测状态是二维的，而隐藏状态有4个。因此我们的“means”参数是4 \times 2的矩阵，而“covars”参数是4 \times 2 \times 2的张量。

建立模型如下：

[![](http://common.cnblogs.com/images/copycode.gif "复制代码")](javascript:void%280%29;)

```
startprob = np.array([0.6, 0.3, 0.1, 0.0
])

#
 The transition matrix, note that there are no transitions possible

#
 between component 1 and 3

transmat = np.array([[0.7, 0.2, 0.0, 0.1
],
                     [
0.3, 0.5, 0.2, 0.0
],
                     [
0.0, 0.3, 0.5, 0.2
],
                     [
0.2, 0.0, 0.2, 0.6
]])

#
 The means of each component

means = np.array([[0.0,  0.0
],
                  [
0.0, 11.0
],
                  [
9.0, 10.0
],
                  [
11.0, -1.0
]])

#
 The covariance of each component

covars = .5 * np.tile(np.identity(2), (4, 1, 1
))


#
 Build an HMM instance and set parameters

model3 = hmm.GaussianHMM(n_components=4, covariance_type=
"
full
"
)


#
 Instead of fitting it from the data, we directly set the estimated

#
 parameters, the means and covariance of the components

model3.startprob_ =
 startprob
model3.transmat_ 
=
 transmat
model3.means_ 
=
 means
model3.covars_ 
= covars
```

[![](http://common.cnblogs.com/images/copycode.gif "复制代码")](javascript:void%280%29;)

注意上面有个参数covariance\_type，取值为"full"意味所有的\mu,\Sigma都需要指定。取值为“spherical”则\Sigma的非对角线元素为0，对角线元素相同。取值为“diag”则\Sigma的非对角线元素为0，对角线元素可以不同，"tied"指所有的隐藏状态对应的观测状态分布使用相同的协方差矩阵\Sigma

我们现在跑一跑HMM问题一解码的过程，由于观测状态是二维的，我们用的三维观测序列， 所以这里的 输入是一个3 \times 2 \times 2的张量，代码如下：

```
seen = np.array([[1.1,2.0],[-1,2.0],[3,7
]])
logprob, state 
= model.decode(seen, algorithm=
"
viterbi
"
)

print
 state
```

输出结果如下：

```
[0 0 1]
```

再看看HMM问题一对数概率的计算：

```
print
 model3.score(seen)
```

输出如下：

```
-41.1211281377
```



