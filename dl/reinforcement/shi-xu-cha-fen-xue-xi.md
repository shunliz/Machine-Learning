时序差分学习\(Temporal-Difference Learning\)结合了动态规划和蒙特卡洛方法，是强化学习的核心思想。

蒙特卡洛的方法是模拟\(或者经历\)一段序列，在序列结束后，根据序列上各个状态的价值，来估计状态价值。

时序差分学习是模拟\(或者经历\)一段序列，每行动一步\(或者几步\)，根据新状态的价值，然后估计执行前的状态价值。

可以认为蒙特卡洛的方法是最大步数的时序差分学习。![](http://statics.2cto.com/js/ckeditor/images/spacer.gif?t=C9G6 "锚点")

DP，MC和TD的区别

DP：已知转移概率p\(s′,r\|s,a\)，Rt+1是精确算出来的，用的vπ\(st+1\)是当前的估计值。

![](https://www.2cto.com/uploadfile/Collfiles/20170607/20170607103511628.png "这里写图片描述")

MC：用多个episode的Gt?近似Gt

TD：Rt+1和vπ\(st+1\)用的都是当前的估计值

![](https://www.2cto.com/uploadfile/Collfiles/20170607/20170607103512630.png "这里写图片描述")

本章介绍的是时序差分学习的单步学习方法。多步学习方法在下一章介绍。主要方法包括：

策略状态价值vπ的时序差分学习方法\(单步\多步\)

策略行动价值qπ的on-policy时序差分学习方法: Sarsa\(单步\多步\)

策略行动价值qπ的off-policy时序差分学习方法: Q-learning\(单步\)

Double Q-learning\(单步\)

策略行动价值qπ的off-policy时序差分学习方法\(带importance sampling\): Sarsa\(多步\)

策略行动价值qπ的off-policy时序差分学习方法\(不带importance sampling\): Tree Backup Algorithm\(多步\)

策略行动价值qπ的off-policy时序差分学习方法:Q\(σ\)\(多步\)

![](http://statics.2cto.com/js/ckeditor/images/spacer.gif?t=C9G6 "锚点")

策略状态价值vπ的时序差分学习方法\(单步\多步\)

单步时序差分学习方法

![](/assets/rl-td1.png)

该算法就通过当前状态的估计与未来估计之间差值来更新状态价值函数的。即R+γV\(S′\)?V\(S\)

相比DP，TD不需要知道p\(s′,r\|s,a\)，Rt+1;相比MC，TD不需要等到序列结束才更新状态值函数，另外MC趋向于拟合样本值，会表现出类似过拟合的性质，TD在估值前会对潜在的Markov过程建模，然后基于该模型估值。例如下面的例子：

![](/assets/rl-td2.png)

多步时序差分学习方法

单步的报酬为：G\(1\)t=Rt+1+γVt\(St+1\)

其中Vt在这里是在时刻t对vπ的估计，而γVt\(St+1\)表示对后面时刻的估计，即γRt+2+γ2Rt+3+?+γT?t?1RT，由此可以得到n步的报酬：

G\(n\)t=Rt+1+γRt+2+?+γn?1Rt+n+γnVt+n?1\(St+n\),n≥1,0≤t

则更新公式为：

Vt+n\(St\)=Vt+n?1\(St\)+α\[G\(n\)t?Vt+n?1\(St\)\]

n-step TD就是利用值函数Vt+n?1来不断估计不可知的报酬Rt+n，则整体框架如下：



