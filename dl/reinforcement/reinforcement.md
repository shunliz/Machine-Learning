除了agent和环境之外，强化学习的要素还包括**策略\(Policy\)**、**奖励\(reward signal\)**、**值函数\(value function\)**、**环境模型\(model\)**，下面对这几种要素进行说明：

1. **策略\(Policy\)**
   ，策略就是一个从当环境状态到行为的映射；
2. **奖励\(reward signal\)**
   ，奖励是agent执行一次行为获得的反馈，强化学习系统的目标是最大化累积的奖励，在不同状态下执行同一个行为可能会得到不同的奖励；
3. **值函数\(value function\)**
   ，一种状态的value为从当前状态出发到停机状态所获得的累积的奖励；
4. **环境模型\(model\)**
   ，agent能够根据环境模型预测环境的行为，采用环境模型的强化学习方法称为基于模型\(model-based\)的方法，不采用环境模型的强化学习方法称为model-free方法。

![](/assets/reinforcementlearning1.png)

强化学习因其注重agent在与环境的直接交互中进行学习而有别于其他学习方

---

增强学习的三大类

* 基于策略的增强学习 Policy Gradients
* 基于最优值的增强学习 Q learning  Sarsa  Deep Q network
* 基于模型的增强学习: model based RL

：

