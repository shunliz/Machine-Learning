# 鲍姆-韦尔奇算法求解HMM参数

---

# 1. HMM模型参数求解概述

HMM模型参数求解根据已知的条件可以分为两种情况。

第一种情况较为简单，就是我们已知D个长度为T的观测序列和对应的隐藏状态序列，即$${(O_1, I_1), (O_2, I_2), ...(O_D, I_D)}$$是已知的，此时我们可以很容易的用最大似然来求解模型参数。

假设样本从隐藏状态$$q_i$$转移到$$q_j$$的频率计数是$$A_{ij}$$,那么状态转移矩阵求得为：$$A = \Big[a_{ij}\Big], \;其中a_{ij} = \frac{A_{ij}}{\sum\limits_{s=1}^{N}A_{is}}$$

假设样本隐藏状态为$$q_j$$且观测状态为$$v_k$$的频率计数是$$B_{jk}$$,那么观测状态概率矩阵为：$$B= \Big[b_{j}(k)\Big], \;其中b_{j}(k) = \frac{B_{jk}}{\sum\limits_{s=1}^{M}B_{js}}$$

假设所有样本中初始隐藏状态为$$q_i$$的频率计数为$$C(i)$$,那么初始概率分布为：$$\Pi = \pi(i) = \frac{C(i)}{\sum\limits_{s=1}^{N}C(s)}$$

可见第一种情况下求解模型还是很简单的。但是在很多时候，我们无法得到HMM样本观察序列对应的隐藏序列，只有D个长度为T的观测序列，即$${(O_1), (O_2), ...(O_D)}$$是已知的，此时我们能不能求出合适的HMM模型参数呢？这就是我们的第二种情况，也是我们本文要讨论的重点。它的解法最常用的是鲍姆-韦尔奇算法，其实就是基于EM算法的求解，只不过鲍姆-韦尔奇算法出现的时代，EM算法还没有被抽象出来，所以我们本文还是说鲍姆-韦尔奇算法。

# 2. 鲍姆-韦尔奇算法原理

鲍姆-韦尔奇算法原理既然使用的就是EM算法的原理，那么我们需要在E步求出联合分布$$P(O,I|\lambda)$$基于条件概率$$P(I|O,\overline{\lambda})$$的期望，其中$$\overline{\lambda}$$为当前的模型参数，然后再M步最大化这个期望，得到更新的模型参数$$\lambda$$。接着不停的进行EM迭代，直到模型参数的值收敛为止。

首先来看看E步，当前模型参数为$$\overline{\lambda}$$, 联合分布$$P(O,I|\lambda)$$基于条件概率$$P(I|O,\overline{\lambda})$$的期望表达式为：$$L(\lambda, \overline{\lambda}) = \sum\limits_{d=1}^D\sum\limits_{I}P(I|O,\overline{\lambda})logP(O,I|\lambda)$$

在M步，我们极大化上式，然后得到更新后的模型参数如下：　$$\overline{\lambda} = arg\;\max_{\lambda}\sum\limits_{d=1}^D\sum\limits_{I}P(I|O,\overline{\lambda})logP(O,I|\lambda)$$

通过不断的E步和M步的迭代，直到$$\overline{\lambda}$$收敛。下面我们来看看鲍姆-韦尔奇算法的推导过程。

# 3. 鲍姆-韦尔奇算法的推导

我们的训练数据为$${(O_1, I_1), (O_2, I_2), ...(O_D, I_D)}$$，其中任意一个观测序列$$O_d = {o_1^{(d)}, o_2^{(d)}, ... o_T^{(d)}}$$,其对应的未知的隐藏状态序列表示为：$$O_d = {i_1^{(d)}, i_2^{(d)}, ... i_T^{(d)}}$$

首先看鲍姆-韦尔奇算法的E步，我们需要先计算联合分布$$P(O,I|\lambda)$$的表达式如下：$$P(O,I|\lambda) = \pi_{i_1}b_{i_1}(o_1)a_{i_1i_2}b_{i_2}(o_2)...a_{i_{T-1}i_T}b_{i_T}(o_T)$$

我们的E步得到的期望表达式为：$$L(\lambda, \overline{\lambda}) = \sum\limits_{d=1}^D\sum\limits_{I}P(I|O,\overline{\lambda})logP(O,I|\lambda)$$

在M步我们要极大化上式。由于$$P(I|O,\overline{\lambda}) = P(I,O|\overline{\lambda})/P(O|\overline{\lambda})$$,而$$P(O|\overline{\lambda})$$是常数，因此我们要极大化的式子等价于：$$\overline{\lambda} = arg\;\max_{\lambda}\sum\limits_{d=1}^D\sum\limits_{I}P(O,I|\overline{\lambda})logP(O,I|\lambda)$$

我们将上面$$P(O,I|\lambda)$$的表达式带入我们的极大化式子，得到的表达式如下：$$\overline{\lambda} = arg\;\max_{\lambda}\sum\limits_{d=1}^D\sum\limits_{I}P(O,I|\overline{\lambda})(log\pi_{i_1} + \sum\limits_{t=1}^{T-1}log\;a_{i_t}a_{i_{t+1}} +  \sum\limits_{t=1}^Tb_{i_t}(o_t))$$

我们的隐藏模型参数$$\lambda =(A,B,\Pi)$$,因此下面我们只需要对上式分别对A,B,$$\Pi$$求导即可得到我们更新的模型参数$$\overline{\lambda}$$

首先我们看看对模型参数$$\Pi$$的求导。由于$$\Pi$$只在上式中括号里的第一部分出现，因此我们对于$$\Pi$$的极大化式子为：$$\overline{\pi_i} = arg\;\max_{\pi_{i_1}} \sum\limits_{d=1}^D\sum\limits_{I}P(O,I|\overline{\lambda})log\pi_{i_1} = arg\;\max_{\pi_{i}} \sum\limits_{d=1}^D\sum\limits_{i=1}^NP(O,i_1^{(d)} =i|\overline{\lambda})log\pi_{i}$$

由于$$\pi_i$$还满足$$\sum\limits_{i=1}^N\pi_i =1$$，因此根据拉格朗日子乘法，我们得到$$\pi_i$$要极大化的拉格朗日函数为：$$arg\;\max_{\pi_{i}}\sum\limits_{d=1}^D\sum\limits_{i=1}^NP(O,i_1^{(d)} =i|\overline{\lambda})log\pi_{i} + \gamma(\sum\limits_{i=1}^N\pi_i -1)$$

其中，$$\gamma$$为拉格朗日系数。上式对$$\pi_i$$求偏导数并令结果为0， 我们得到：$$\sum\limits_{d=1}^DP(O,i_1^{(d)} =i|\overline{\lambda}) + \gamma\pi_i = 0$$

令i分别等于从1到N，从上式可以得到N个式子，对这N个式子求和可得：$$\sum\limits_{d=1}^DP(O|\overline{\lambda}) + \gamma = 0$$

从上两式消去$$\gamma$$,得到$$\pi_i$$的表达式为：$$\pi_i =\frac{\sum\limits_{d=1}^DP(O,i_1^{(d)} =i|\overline{\lambda})}{\sum\limits_{d=1}^DP(O|\overline{\lambda})} = \frac{\sum\limits_{d=1}^DP(O,i_1^{(d)} =i|\overline{\lambda})}{DP(O|\overline{\lambda})} = \frac{\sum\limits_{d=1}^DP(i_1^{(d)} =i|O, \overline{\lambda})}{D} =  \frac{\sum\limits_{d=1}^DP(i_1^{(d)} =i|O^{(d)}, \overline{\lambda})}{D}$$

利用我们在[隐马尔科夫模型HMM（二）前向后向算法评估观察序列概率](/ml/hmm/hmm-forward-backward.md)里第二节中前向概率的定义可得：$$P(i_1^{(d)} =i|O^{(d)}, \overline{\lambda}) = \gamma_1^{(d)}(i)$$

因此最终我们在M步$$\pi_i$$的迭代公式为：$$\pi_i =  \frac{\sum\limits_{d=1}^D\gamma_1^{(d)}(i)}{D}$$

现在我们来看看A的迭代公式求法。方法和$$\Pi$$的类似。由于A只在最大化函数式中括号里的第二部分出现，而这部分式子可以整理为：$$\sum\limits_{d=1}^D\sum\limits_{I}\sum\limits_{t=1}^{T-1}P(O,I|\overline{\lambda})log\;a_{i_t}a_{i_{t+1}} = \sum\limits_{d=1}^D\sum\limits_{i=1}^N\sum\limits_{j=1}^N\sum\limits_{t=1}^{T-1}P(O,i_t^{(d)} = i, i_{t+1}^{(d)} = j|\overline{\lambda})log\;a_{ij}$$

由于$$a_{ij}$$还满足$$\sum\limits_{j=1}^Na_{ij} =1$$。和求解$$\pi_i$$类似，我们可以用拉格朗日子乘法并对$$a_{ij}$$求导，并令结果为0，可以得到$$a_{ij}$$的迭代表达式为：$$a_{ij} = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}P(O^{(d)}, i_t^{(d)} = i, i_{t+1}^{(d)} = j|\overline{\lambda})}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}P(O^{(d)}, i_t^{(d)} = i|\overline{\lambda})}$$

利用[隐马尔科夫模型HMM（二）前向后向算法评估观察序列概率](/ml/hmm/hmm-forward-backward.md)里第二节中前向概率的定义和第五节$$\xi_t(i,j)$$的定义可得们在M步$$a_{ij}$$的迭代公式为：$$a_{ij} = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}\xi_t^{(d)}(i,j)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}\gamma_t^{(d)}(i)}$$

现在我们来看看B的迭代公式求法。方法和$$\Pi$$的类似。由于B只在最大化函数式中括号里的第三部分出现，而这部分式子可以整理为：$$\sum\limits_{d=1}^D\sum\limits_{I}\sum\limits_{t=1}^{T}P(O,I|\overline{\lambda})log\;b_{i_t}(o_t) = \sum\limits_{d=1}^D\sum\limits_{j=1}^N\sum\limits_{t=1}^{T}P(O,i_t^{(d)} = j|\overline{\lambda})log\;b_{j}(o_t)$$

由于$$b_{j}(o_t)$$还满足$$\sum\limits_{k=1}^Mb_{j}(o_t =v_k) =1$$。和求解$$\pi_i$$类似，我们可以用拉格朗日子乘法并对$$b_{j}(k)$$求导，并令结果为0，得到$$b_{j}(k)$$的迭代表达式为：$$b_{j}(k) = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T}P(O,i_t^{(d)} = j|\overline{\lambda})I(o_t^{(d)}=v_k)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T}P(O,i_t^{(d)} = j|\overline{\lambda})}$$

其中$$I(o_t^{(d)}=v_k)$$当且仅当$$o_t^{(d)}=v_k$$时为1，否则为0. 利用[隐马尔科夫模型HMM（二）前向后向算法评估观察序列概率](/ml/hmm/hmm-forward-backward.md)里第二节中前向概率的定义可得$$b_{j}(o_t)$$的最终表达式为：$$b_{j}(k) = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1, o_t^{(d)}=v_k}^{T}\gamma_t^{(d)}(i)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T}\gamma_t^{(d)}(i)}$$

有了$$\pi_i, a_{ij},b_{j}(k)$$的迭代公式，我们就可以迭代求解HMM模型参数了。

# 4. 鲍姆-韦尔奇算法流程总结

这里我们概括总结下鲍姆-韦尔奇算法的流程。

输入：D个观测序列样本$${(O_1), (O_2), ...(O_D)}$$

输出：HMM模型参数

1\)随机初始化所有的$$\pi_i, a_{ij},b_{j}(k)$$

2\) 对于每个样本d = 1,2,...D，用前向后向算法计算$$\gamma_t^{(d)}(i)，\xi_t^{(d)}(i,j), t =1,2...T$$

3\)  更新模型参数：

$$\pi_i =  \frac{\sum\limits_{d=1}^D\gamma_1^{(d)}(i)}{D}$$

$$a_{ij} = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}\xi_t^{(d)}(i,j)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T-1}\gamma_t^{(d)}(i)}$$

$$b_{j}(k) = \frac{\sum\limits_{d=1}^D\sum\limits_{t=1, o_t^{(d)}=v_k}^{T}\gamma_t^{(d)}(i)}{\sum\limits_{d=1}^D\sum\limits_{t=1}^{T}\gamma_t^{(d)}(i)}$$

4\) 如果$$\pi_i, a_{ij},b_{j}(k)$$的值已经收敛，则算法结束，否则回到第2）步继续迭代。

