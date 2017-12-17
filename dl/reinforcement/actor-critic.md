Actor 和 Critic, 他们都能用不同的神经网络来代替 . 在 Policy Gradients 的影片中提到过, 现实中的奖惩会左右 Actor 的更新情况. Policy Gradients 也是靠着这个来获取适宜的更新. 那么何时会有奖惩这种信息能不能被学习呢? 这看起来不就是 以值为基础的强化学习方法做过的事吗. 那我们就拿一个 Critic 去学习这些奖惩机制, 学习完了以后. 由 Actor 来指手画脚, 由 Critic 来告诉 Actor 你的那些指手画脚哪些指得好, 哪些指得差, Critic 通过学习环境和奖励之间的关系, 能看到现在所处状态的潜在奖励, 所以用它来指点 Actor 便能使 Actor 每一步都在更新, 如果使用单纯的 Policy Gradients, Actor 只能等到回合结束才能开始更新.



**一句话概括 Actor Critic 方法**:

结合了 Policy Gradient \(Actor\) 和 Function Approximation \(Critic\) 的方法.`Actor`基于概率选行为,`Critic`基于`Actor`的行为评判行为的得分,`Actor`根据`Critic`的评分修改选行为的概率.

**Actor Critic 方法的优势**: 可以进行单步更新, 比传统的 Policy Gradient 要快.

**Actor Critic 方法的劣势**: 取决于 Critic 的价值判断, 但是 Critic 难收敛, 再加上 Actor 的更新, 就更难收敛. 为了解决收敛问题, Google Deepmind 提出了`Actor Critic`升级版`Deep Deterministic Policy Gradient`. 后者融合了 DQN 的优势, 解决了收敛难的问题. 我们之后也会要讲到[Deep Deterministic Policy Gradient](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-2-DDPG/). 不过那个是要以`Actor Critic`为基础, 懂了`Actor Critic`, 后面那个就好懂了.



