# 鲍姆-韦尔奇算法求解HMM参数

---

# 1. HMM模型参数求解概述

　　　　HMM模型参数求解根据已知的条件可以分为两种情况。

　　　　第一种情况较为简单，就是我们已知D个长度为T的观测序列和对应的隐藏状态序列，即\{\(O\_1, I\_1\), \(O\_2, I\_2\), ...\(O\_D, I\_D\)\}是已知的，此时我们可以很容易的用最大似然来求解模型参数。

　　　　假设样本从隐藏状态q\_i转移到q\_j的频率计数是A\_{ij},那么状态转移矩阵求得为：A = \Big\[a\_{ij}\Big\], \;其中a\_{ij} = \frac{A\_{ij}}{\sum\limits\_{s=1}^{N}A\_{is}}

　　　　假设样本隐藏状态为q\_j且观测状态为v\_k的频率计数是B\_{jk},那么观测状态概率矩阵为：B= \Big\[b\_{j}\(k\)\Big\], \;其中b\_{j}\(k\) = \frac{B\_{jk}}{\sum\limits\_{s=1}^{M}B\_{js}}

　　　　假设所有样本中初始隐藏状态为q\_i的频率计数为C\(i\),那么初始概率分布为：\Pi = \pi\(i\) = \frac{C\(i\)}{\sum\limits\_{s=1}^{N}C\(s\)}

　　　　可见第一种情况下求解模型还是很简单的。但是在很多时候，我们无法得到HMM样本观察序列对应的隐藏序列，只有D个长度为T的观测序列，即\{\(O\_1\), \(O\_2\), ...\(O\_D\)\}是已知的，此时我们能不能求出合适的HMM模型参数呢？这就是我们的第二种情况，也是我们本文要讨论的重点。它的解法最常用的是鲍姆-韦尔奇算法，其实就是基于EM算法的求解，只不过鲍姆-韦尔奇算法出现的时代，EM算法还没有被抽象出来，所以我们本文还是说鲍姆-韦尔奇算法。

# 2. 鲍姆-韦尔奇算法原理

　　　　鲍姆-韦尔奇算法原理既然使用的就是EM算法的原理，那么我们需要在E步求出联合分布P\(O,I\|\lambda\)基于条件概率P\(I\|O,\overline{\lambda}\)的期望，其中\overline{\lambda}为当前的模型参数，然后再M步最大化这个期望，得到更新的模型参数\lambda。接着不停的进行EM迭代，直到模型参数的值收敛为止。

　　　　首先来看看E步，当前模型参数为\overline{\lambda}, 联合分布P\(O,I\|\lambda\)基于条件概率P\(I\|O,\overline{\lambda}\)的期望表达式为：L\(\lambda, \overline{\lambda}\) = \sum\limits\_{d=1}^D\sum\limits\_{I}P\(I\|O,\overline{\lambda}\)logP\(O,I\|\lambda\)

　　　　在M步，我们极大化上式，然后得到更新后的模型参数如下：　\overline{\lambda} = arg\;\max\_{\lambda}\sum\limits\_{d=1}^D\sum\limits\_{I}P\(I\|O,\overline{\lambda}\)logP\(O,I\|\lambda\)

　　　　通过不断的E步和M步的迭代，直到\overline{\lambda}收敛。下面我们来看看鲍姆-韦尔奇算法的推导过程。

# 3. 鲍姆-韦尔奇算法的推导

　　　　我们的训练数据为\{\(O\_1, I\_1\), \(O\_2, I\_2\), ...\(O\_D, I\_D\)\}，其中任意一个观测序列O\_d = \{o\_1^{\(d\)}, o\_2^{\(d\)}, ... o\_T^{\(d\)}\},其对应的未知的隐藏状态序列表示为：O\_d = \{i\_1^{\(d\)}, i\_2^{\(d\)}, ... i\_T^{\(d\)}\}

　　　　首先看鲍姆-韦尔奇算法的E步，我们需要先计算联合分布P\(O,I\|\lambda\)的表达式如下：P\(O,I\|\lambda\) = \pi\_{i\_1}b\_{i\_1}\(o\_1\)a\_{i\_1i\_2}b\_{i\_2}\(o\_2\)...a\_{i\_{T-1}i\_T}b\_{i\_T}\(o\_T\)

　　　　我们的E步得到的期望表达式为：L\(\lambda, \overline{\lambda}\) = \sum\limits\_{d=1}^D\sum\limits\_{I}P\(I\|O,\overline{\lambda}\)logP\(O,I\|\lambda\)

　　　　在M步我们要极大化上式。由于P\(I\|O,\overline{\lambda}\) = P\(I,O\|\overline{\lambda}\)/P\(O\|\overline{\lambda}\),而P\(O\|\overline{\lambda}\)是常数，因此我们要极大化的式子等价于：\overline{\lambda} = arg\;\max\_{\lambda}\sum\limits\_{d=1}^D\sum\limits\_{I}P\(O,I\|\overline{\lambda}\)logP\(O,I\|\lambda\)

　　　　我们将上面P\(O,I\|\lambda\)的表达式带入我们的极大化式子，得到的表达式如下：\overline{\lambda} = arg\;\max\_{\lambda}\sum\limits\_{d=1}^D\sum\limits\_{I}P\(O,I\|\overline{\lambda}\)\(log\pi\_{i\_1} + \sum\limits\_{t=1}^{T-1}log\;a\_{i\_t}a\_{i\_{t+1}} +  \sum\limits\_{t=1}^Tb\_{i\_t}\(o\_t\)\)

　　　　我们的隐藏模型参数\lambda =\(A,B,\Pi\),因此下面我们只需要对上式分别对A,B,\Pi求导即可得到我们更新的模型参数\overline{\lambda}



　　　　首先我们看看对模型参数\Pi的求导。由于\Pi只在上式中括号里的第一部分出现，因此我们对于\Pi的极大化式子为：\overline{\pi\_i} = arg\;\max\_{\pi\_{i\_1}} \sum\limits\_{d=1}^D\sum\limits\_{I}P\(O,I\|\overline{\lambda}\)log\pi\_{i\_1} = arg\;\max\_{\pi\_{i}} \sum\limits\_{d=1}^D\sum\limits\_{i=1}^NP\(O,i\_1^{\(d\)} =i\|\overline{\lambda}\)log\pi\_{i}

　　　　由于\pi\_i还满足\sum\limits\_{i=1}^N\pi\_i =1，因此根据拉格朗日子乘法，我们得到\pi\_i要极大化的拉格朗日函数为：arg\;\max\_{\pi\_{i}}\sum\limits\_{d=1}^D\sum\limits\_{i=1}^NP\(O,i\_1^{\(d\)} =i\|\overline{\lambda}\)log\pi\_{i} + \gamma\(\sum\limits\_{i=1}^N\pi\_i -1\)

　　　　其中，\gamma为拉格朗日系数。上式对\pi\_i求偏导数并令结果为0， 我们得到：\sum\limits\_{d=1}^DP\(O,i\_1^{\(d\)} =i\|\overline{\lambda}\) + \gamma\pi\_i = 0

　　　　令i分别等于从1到N，从上式可以得到N个式子，对这N个式子求和可得：\sum\limits\_{d=1}^DP\(O\|\overline{\lambda}\) + \gamma = 0

　　　　从上两式消去\gamma,得到\pi\_i的表达式为：\pi\_i =\frac{\sum\limits\_{d=1}^DP\(O,i\_1^{\(d\)} =i\|\overline{\lambda}\)}{\sum\limits\_{d=1}^DP\(O\|\overline{\lambda}\)} = \frac{\sum\limits\_{d=1}^DP\(O,i\_1^{\(d\)} =i\|\overline{\lambda}\)}{DP\(O\|\overline{\lambda}\)} = \frac{\sum\limits\_{d=1}^DP\(i\_1^{\(d\)} =i\|O, \overline{\lambda}\)}{D} =  \frac{\sum\limits\_{d=1}^DP\(i\_1^{\(d\)} =i\|O^{\(d\)}, \overline{\lambda}\)}{D}

　　　　利用我们在[隐马尔科夫模型HMM（二）前向后向算法评估观察序列概率](http://www.cnblogs.com/pinard/p/6955871.html)里第二节中前向概率的定义可得：P\(i\_1^{\(d\)} =i\|O^{\(d\)}, \overline{\lambda}\) = \gamma\_1^{\(d\)}\(i\)

　　　　因此最终我们在M步\pi\_i的迭代公式为：\pi\_i =  \frac{\sum\limits\_{d=1}^D\gamma\_1^{\(d\)}\(i\)}{D}



　　　　现在我们来看看A的迭代公式求法。方法和\Pi的类似。由于A只在最大化函数式中括号里的第二部分出现，而这部分式子可以整理为：\sum\limits\_{d=1}^D\sum\limits\_{I}\sum\limits\_{t=1}^{T-1}P\(O,I\|\overline{\lambda}\)log\;a\_{i\_t}a\_{i\_{t+1}} = \sum\limits\_{d=1}^D\sum\limits\_{i=1}^N\sum\limits\_{j=1}^N\sum\limits\_{t=1}^{T-1}P\(O,i\_t^{\(d\)} = i, i\_{t+1}^{\(d\)} = j\|\overline{\lambda}\)log\;a\_{ij}

　　　　由于a\_{ij}还满足\sum\limits\_{j=1}^Na\_{ij} =1。和求解\pi\_i类似，我们可以用拉格朗日子乘法并对a\_{ij}求导，并令结果为0，可以得到a\_{ij}的迭代表达式为：a\_{ij} = \frac{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T-1}P\(O^{\(d\)}, i\_t^{\(d\)} = i, i\_{t+1}^{\(d\)} = j\|\overline{\lambda}\)}{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T-1}P\(O^{\(d\)}, i\_t^{\(d\)} = i\|\overline{\lambda}\)}

　　　　利用[隐马尔科夫模型HMM（二）前向后向算法评估观察序列概率](http://www.cnblogs.com/pinard/p/6955871.html)里第二节中前向概率的定义和第五节\xi\_t\(i,j\)的定义可得们在M步a\_{ij}的迭代公式为：a\_{ij} = \frac{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T-1}\xi\_t^{\(d\)}\(i,j\)}{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T-1}\gamma\_t^{\(d\)}\(i\)}



　　　　现在我们来看看B的迭代公式求法。方法和\Pi的类似。由于B只在最大化函数式中括号里的第三部分出现，而这部分式子可以整理为：\sum\limits\_{d=1}^D\sum\limits\_{I}\sum\limits\_{t=1}^{T}P\(O,I\|\overline{\lambda}\)log\;b\_{i\_t}\(o\_t\) = \sum\limits\_{d=1}^D\sum\limits\_{j=1}^N\sum\limits\_{t=1}^{T}P\(O,i\_t^{\(d\)} = j\|\overline{\lambda}\)log\;b\_{j}\(o\_t\)

　　　　由于b\_{j}\(o\_t\)还满足\sum\limits\_{k=1}^Mb\_{j}\(o\_t =v\_k\) =1。和求解\pi\_i类似，我们可以用拉格朗日子乘法并对b\_{j}\(k\)求导，并令结果为0，得到b\_{j}\(k\)的迭代表达式为：b\_{j}\(k\) = \frac{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T}P\(O,i\_t^{\(d\)} = j\|\overline{\lambda}\)I\(o\_t^{\(d\)}=v\_k\)}{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T}P\(O,i\_t^{\(d\)} = j\|\overline{\lambda}\)}

　　　　其中I\(o\_t^{\(d\)}=v\_k\)当且仅当o\_t^{\(d\)}=v\_k时为1，否则为0. 利用[隐马尔科夫模型HMM（二）前向后向算法评估观察序列概率](http://www.cnblogs.com/pinard/p/6955871.html)里第二节中前向概率的定义可得b\_{j}\(o\_t\)的最终表达式为：b\_{j}\(k\) = \frac{\sum\limits\_{d=1}^D\sum\limits\_{t=1, o\_t^{\(d\)}=v\_k}^{T}\gamma\_t^{\(d\)}\(i\)}{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T}\gamma\_t^{\(d\)}\(i\)}

　　　　有了\pi\_i, a\_{ij},b\_{j}\(k\)的迭代公式，我们就可以迭代求解HMM模型参数了。

# 4. 鲍姆-韦尔奇算法流程总结

　　　　这里我们概括总结下鲍姆-韦尔奇算法的流程。

　　　　输入：D个观测序列样本\{\(O\_1\), \(O\_2\), ...\(O\_D\)\}

　　　　输出：HMM模型参数

　　　　1\)随机初始化所有的\pi\_i, a\_{ij},b\_{j}\(k\)

　　　　2\) 对于每个样本d = 1,2,...D，用前向后向算法计算\gamma\_t^{\(d\)}\(i\)，\xi\_t^{\(d\)}\(i,j\), t =1,2...T

　　　　3\)  更新模型参数：

\pi\_i =  \frac{\sum\limits\_{d=1}^D\gamma\_1^{\(d\)}\(i\)}{D}

a\_{ij} = \frac{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T-1}\xi\_t^{\(d\)}\(i,j\)}{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T-1}\gamma\_t^{\(d\)}\(i\)}

b\_{j}\(k\) = \frac{\sum\limits\_{d=1}^D\sum\limits\_{t=1, o\_t^{\(d\)}=v\_k}^{T}\gamma\_t^{\(d\)}\(i\)}{\sum\limits\_{d=1}^D\sum\limits\_{t=1}^{T}\gamma\_t^{\(d\)}\(i\)}

　　　　4\) 如果\pi\_i, a\_{ij},b\_{j}\(k\)的值已经收敛，则算法结束，否则回到第2）步继续迭代。

