本文简明讲述GMM-HMM在语音识别上的原理，建模和测试过程。这篇blog只回答三个问题：

1. 什么是[Hidden Markov Model](http://en.wikipedia.org/wiki/Hidden_Markov_models)？

 HMM要解决的三个问题:

 1\) Likelihood

 2\) Decoding

 3\) Training

2. GMM是神马？怎样用GMM求某一音素（phoneme）的概率？



3. GMM+HMM大法解决语音识别

 3.1 识别

 3.2 训练

 3.2.1 Training the params of GMM

 3.2.2 Training the params of HMM

  




  


首先声明我是做视觉的不是做语音的，迫于\*\*需要24小时速成语音。上网查GMM-HMM资料中文几乎为零，英文也大多是paper。苦苦追寻终于貌似搞懂了GMM-HMM，感谢语音组老夏（[http://weibo.com/ibillxia](http://weibo.com/ibillxia)）提供资料给予指导。本文结合最简明的概括还有自己一些理解应运而生，如有错误望批评指正。

  


====================================================================

  


  


1. 什么是[Hidden Markov Model](http://en.wikipedia.org/wiki/Hidden_Markov_models)？

![](http://img.blog.csdn.net/20140528174242250?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


ANS：一个有隐节点（unobservable）和可见节点（visible）的马尔科夫过程（见[详解](http://blog.csdn.net/abcjennifer/article/details/25908495)）。

隐节点表示状态，可见节点表示我们听到的语音或者看到的时序信号。

最开始时，我们指定这个HMM的结构，训练HMM模型时：给定n个时序信号y1...yT（训练样本）, 用MLE（typicallyimplemented in EM） 估计参数：

1. N个状态的初始概率

2. 状态转移概率a

3. 输出概率b

--------------



* 在语音处理中，一个word由若干phoneme（音素）组成；
* 每个HMM对应于一个word或者音素（phoneme）
* 一个word表示成若干states，每个state表示为一个音素

  


用HMM需要解决3个问题：

1）.Likelihood: 一个HMM生成一串observation序列x的概率&lt; the Forward algorithm&gt;

![](http://img.blog.csdn.net/20140530151854546?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中，αt（sj）表示HMM在时刻t处于状态j，且observation = {x1,...,xt}的概率![](http://img.blog.csdn.net/20140530152949593?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)，

aij是状态i到状态j的转移概率，

bj（xt）表示在状态j的时候生成xt的概率，

  


  


  


  


  


  


2）.Decoding: 给定一串observation序列x，找出最可能从属的HMM状态序列&lt; the Viterbi algorithm&gt;



在实际计算中会做剪枝，不是计算每个可能state序列的probability，而是用Viterbi approximation：

从时刻1：t，只记录转移概率最大的state和概率。

记Vt\(si\)为从时刻t-1的所有状态转移到时刻t时状态为j的最大概率：![](http://img.blog.csdn.net/20140530155625171?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

记![](http://img.blog.csdn.net/20140530154949078?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)为：从时刻t-1的哪个状态转移到时刻t时状态为j的概率最大；

进行Viterbi approximation过程如下：

![](http://img.blog.csdn.net/20140530155945437?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

  


然后根据记录的最可能转移状态序列![](http://img.blog.csdn.net/20140530154949078?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)进行回溯：

![](http://img.blog.csdn.net/20140530160136578?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


  


  


  


3）.Training: 给定一个observation序列x，训练出HMM参数λ = {aij, bij}  the EM \(Forward-Backward\) algorithm

这部分我们放到“3. GMM+HMM大法解决语音识别”中和GMM的training一起讲

  




  


  


  


  


  


---------------------------------------------------------------------

  


2. GMM是神马？怎样用GMM求某一音素（phoneme）的概率？

2.1 简单理解混合高斯模型就是几个高斯的叠加。。。e.g. k=3

![](http://img.blog.csdn.net/20140528180736578?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


![](http://img.blog.csdn.net/20140530134729015?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


fig2. GMM illustration and the probability of x  


  


  


2.2 GMM for state sequence 

每个state有一个GMM，包含k个高斯模型参数。如”hi“（k=3）：

PS：sil表示silence（静音）

![](http://img.blog.csdn.net/20140528200425421?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

fig3. use GMM to estimate the probability of a state sequence given observation {o1, o2, o3}

  


其中，每个GMM有一些参数，就是我们要train的输出概率参数

![](http://img.blog.csdn.net/20140528200531906?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


fig4. parameters of a GMM

怎么求呢？和KMeans类似，如果已知每个点x^n属于某每类 j 的概率p\(j\|x^n\)，则可以估计其参数:

![](http://img.blog.csdn.net/20140530135251546?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) ， 其中 ![](http://img.blog.csdn.net/20140530135311953?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


  


只要已知了这些参数，我们就可以在predict（识别）时在给定input sequence的情况下，计算出一串状态转移的概率。如上图要计算的state sequence 1-&gt;2-&gt;2概率：

![](http://img.blog.csdn.net/20140528201041078?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


fig5. probability of S1-&gt;S2-&gt;S3 given o1-&gt;o2-&gt;o3

  


  


  


  


  


  


  
---------------------------------------------------------------------  


3. GMM+HMM大法解决语音识别

&lt;!--识别--&gt;  


我们获得observation是语音waveform, 以下是一个词识别全过程：

1\). 将waveform切成等长frames，对每个frame提取特征（e.g. MFCC）, 

2\).对每个frame的特征跑GMM，得到每个frame\(o\_i\)属于每个状态的概率b\_state\(o\_i\)

  


![](http://img.blog.csdn.net/20140528203714828?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


fig6. complete process from speech frames to a state sequence

  


3\). 根据每个单词的HMM状态转移概率a计算每个状态sequence生成该frame的概率; 哪个词的HMM 序列跑出来概率最大，就判断这段语音属于该词

  


宏观图：

![](http://img.blog.csdn.net/20140528175313171?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


fig7. Speech recognition, a big framework

\(from Encyclopedia of Information Systems, 2002\)  


  


  


  


&lt;!--训练--&gt;

好了，上面说了怎么做识别。那么我们怎样训练这个模型以得到每个GMM的参数和HMM的转移概率什么的呢？

  


  


  


①Training the params of GMM

GMM参数：高斯分布参数：![](http://img.blog.csdn.net/20140530185018734?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

从上面fig4下面的公式我们已经可以看出来想求参数必须要知道P\(j\|x\),即，x属于第j个高斯的概率。怎么求捏？

![](http://img.blog.csdn.net/20140530141637656?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


fig8. bayesian formula of P\( j \| x \)

根据上图 P（j \| x）, 我们需要求P\(x\|j\)和P（j）去估计P\(j\|x\). 

这里由于P\(x\|j\)和P（j）都不知道，需要用EM算法迭代估计以最大化P\(x\) = P\(x1\)\*p\(x2\)\*...\*P\(xn\)：

A. 初始化（可以用kmeans）得到P\(j\)

B. 迭代

    E（estimate）-step: 根据当前参数 \(means, variances, mixing parameters\)估计P\(j\|x\)

    M（maximization）-step: 根据当前P\(j\|x\) 计算GMM参数（根据fig4 下面的公式：）

![](http://img.blog.csdn.net/20140530135251546?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) ， 其中 ![](http://img.blog.csdn.net/20140530135311953?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


  


  


  


  


②Training the params of HMM

前面已经有了GMM的training过程。在这一步，我们的目标是：从observation序列中估计HMM参数λ；

假设状态-&gt;observation服从单核高斯概率分布：![](http://img.blog.csdn.net/20140530162550421?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)，则λ由两部分组成：  


  


![](http://img.blog.csdn.net/20140530195145953?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


  


HMM训练过程：迭代



    E（estimate）-step: 给定observation序列，估计时刻t处于状态sj的概率 ![](http://img.blog.csdn.net/20140530185647156?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

    M（maximization）-step: 根据![](http://img.blog.csdn.net/20140530185647156?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)重新估计HMM参数aij. 

其中，

  


E-step: 给定observation序列，估计时刻t处于状态sj的概率 ![](http://img.blog.csdn.net/20140530185647156?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

为了估计![](http://img.blog.csdn.net/20140530185647156?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast), 定义![](http://img.blog.csdn.net/20140530191032625?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast): t时刻处于状态sj的话，t时刻未来observation的概率。即![](http://img.blog.csdn.net/20140530191206000?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

这个可以递归计算：β\_t（si）=从状态 si 转移到其他状态 sj 的概率aij\*状态 i 下观测到x\_{t+1}的概率bi\(x\_{t+1}\)\*t时刻处于状态sj的话{t+1}后observation概率β\_{t+1}（sj）

即：

![](http://img.blog.csdn.net/20140530191353765?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


定义刚才的![](http://img.blog.csdn.net/20140530185647156?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)为state occupation probability，表示给定observation序列，时刻t处于状态sj的概率P\(S\(t\)=sj \| X,λ\)。根据贝叶斯公式p\(A\|B,C\) = P\(A,B\|C\)/P\(B\|C\)，有：

  


![](http://img.blog.csdn.net/20140530194138937?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  




由于分子p\(A,B\|C\)为

![](http://img.blog.csdn.net/20140530193757812?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


其中，αt（sj）表示HMM在时刻t处于状态j，且observation = {x1,...,xt}的概率![](http://img.blog.csdn.net/20140530152949593?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)；  


![](http://img.blog.csdn.net/20140530191032625?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast): t时刻处于状态sj的话，t时刻未来observation的概率；  


且![](http://img.blog.csdn.net/20140530193617734?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

finally, 带入![](http://img.blog.csdn.net/20140530185647156?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)的定义式有：



![](http://img.blog.csdn.net/20140530194816484?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  




好，终于搞定！对应上面的E-step目标，只要给定了observation和当前HMM参数 λ，我们就可以估计![](http://img.blog.csdn.net/20140530185647156?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)了对吧 \(\*^\_\_^\*\) 

  


  


  


  


  


M-step：根据![](http://img.blog.csdn.net/20140530185647156?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)重新估计HMM参数λ：  


对于λ中高斯参数部分，和GMM的M-step是一样一样的（只不过这里写成向量形式）：

![](http://img.blog.csdn.net/20140530200004781?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


对于λ中的状态转移概率aij, 定义C\(Si-&gt;Sj\)为从状态Si转到Sj的次数，有

![](http://img.blog.csdn.net/20140530200136921?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


实际计算时，定义每一时刻的转移概率![](http://img.blog.csdn.net/20140530200404828?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)为时刻t从si-&gt;sj的概率：

![](http://img.blog.csdn.net/20140530200424640?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


那么就有：

![](http://img.blog.csdn.net/20140530200615750?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


把HMM的EM迭代过程和要求的参数写专业点，就是这样的：

![](http://img.blog.csdn.net/20140530200730218?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)  


PS：这个训练HMM的算法叫 Forward-Backward algorithm。

  


  


  


  


一个很好的reference：[点击打开链接](http://www.inf.ed.ac.uk/teaching/courses/asr/2012-13/asr03-hmmgmm-4up.pdf)

