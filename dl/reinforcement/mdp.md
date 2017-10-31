马尔可夫决策过程正式的描述了增强学习所处的环境，在这个环境中，所有都是可观测的。所有的增强学习都可以被转化为MDP。

* 连续MDP的最优控制过程\(Optimal control\)
* Partially observable problems也可以转化为MDP
* 多臂赌博机问题

### 马尔可夫特性 {#马尔可夫特性}

意味着未来的状态只与现在所处的状态有关。过去的history都可以丢弃。

![](/assets/markv-mdp1.png)

### 状态转移矩阵 {#状态转移矩阵}

这里矩阵代表状态转移时候的矩阵，每一行sum为1。

![](/assets/markv-transfer-matrix.png)

### 马尔可夫过程\(Markov Process\) {#马尔可夫过程markov-process}

马尔可夫过程是一个二元组，是一个无记忆的随机过程，是一连串的状态序列具有马尔可夫特性。给出Definition.  
![](/assets/markv-mdp2.png)

S是状态的集合，P是转移矩阵。

## 马尔可夫奖励过程\(Markov Reward Process\) {#马尔可夫奖励过程markov-reward-process}

MRP是带有价值的马尔科夫链，相对于马尔可夫过程，添加了Reward函数，和折现因子γ。给出MRP的定义。

![](/assets/markv-mrp1.png)

