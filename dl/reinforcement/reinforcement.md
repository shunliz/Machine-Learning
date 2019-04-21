# 强化学习关键概念

强化学习通常用来突破一些AI问题，比如下围棋的[AlphaGo](https://deepmind.com/research/alphago/)和[Dota](https://blog.openai.com/openai-five/)人机大战。

![](/assets/reinforcementlearning1.png)

强化学习的两个主要角色就是agent和environment. environment是和agent存在的场所并和agent交互。每一次交互，agent都会收到一个environment的观察值, 然后决定怎么做。当agent做出响应后，环境跟着改变， 同时也可能改变它自己。

agent同时也会从evironment收到一个奖励值， 这个奖励值表明当前的environment有多好或者有多坏。agent的目标就是最大化累计的奖励。强化学习就是agent学习获取最大化回报的学习方法。

谈到强化学习，必须介绍几个相关概念：

* 状态和观测值
* 动作空间
* 策略
* trajectories
* different formulations of return,
* 强化学习问题
* 价值函数



**状态和观测值**

状态是环境信息的完整描述。没有状态不包含的环境信息。

观测值是部分环境信息，部分信息可能不包含在观测值中。



**动作空间**

不同的enironment可以有不同的动作，所有可以施加到environment的动作构成动作空间。

有些动作空间是连续的，有些动作空间是离散的。



**策略**

策略是agent采取动作的规则，可以是确定，表示为：

$$a_t=\mu(s_t)$$

如果是随机的，一般表示为：

$$a_t=\pi(.|s_t)$$

强化学习中我们一般处理参数化的策略。策略的输出是一些包含参数的可计算函数， 我们通过优化算法来调整这些参数，从而改变agent的行为。这时我们通常表示为下边公式：

$$a_t=\mu_\theta(s_t)$$

$$a_t=\pi_|\theta(.|s_t)$$

![](/assets/reinforcementlearning4.png)

![](/assets/reinforcementlearing5.png)

![](/assets/reinforcementlearning6.png)![](/assets/reinforementlearing7.png)

强化学习因其注重agent在与环境的直接交互中进行学习而有别于其他学习方

---

增强学习的三大类

* 基于策略的增强学习 Policy Gradients
* 基于最优值的增强学习 Q learning  Sarsa  Deep Q network
* 基于模型的增强学习: model based RL

：

| [A2C / A3C](https://arxiv.org/abs/1602.01783)\(Asynchronous Advantage Actor-Critic\): Mnih et al, 2016 |
| :--- |


| [PPO](https://arxiv.org/abs/1707.06347)\(Proximal Policy Optimization\): Schulman et al, 2017 |
| :--- |


| [TRPO](https://arxiv.org/abs/1502.05477)\(Trust Region Policy Optimization\): Schulman et al, 2015 |
| :--- |


| [DDPG](https://arxiv.org/abs/1509.02971)\(Deep Deterministic Policy Gradient\): Lillicrap et al, 2015 |
| :--- |


| [TD3](https://arxiv.org/abs/1802.09477)\(Twin Delayed DDPG\): Fujimoto et al, 2018 |
| :--- |


| [SAC](https://arxiv.org/abs/1801.01290)\(Soft Actor-Critic\): Haarnoja et al, 2018 |
| :--- |


| [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)\(Deep Q-Networks\): Mnih et al, 2013 |
| :--- |


| [C51](https://arxiv.org/abs/1707.06887)\(Categorical 51-Atom DQN\): Bellemare et al, 2017 |
| :--- |


| [QR-DQN](https://arxiv.org/abs/1710.10044)\(Quantile Regression DQN\): Dabney et al, 2017 |
| :--- |


| [HER](https://arxiv.org/abs/1707.01495)\(Hindsight Experience Replay\): Andrychowicz et al, 2017 |
| :--- |


| [World Models](https://worldmodels.github.io/): Ha and Schmidhuber, 2018 |
| :--- |


| [I2A](https://arxiv.org/abs/1707.06203)\(Imagination-Augmented Agents\): Weber et al, 2017 |
| :--- |


| [MBMF](https://sites.google.com/view/mbmf)\(Model-Based RL with Model-Free Fine-Tuning\): Nagabandi et al, 2017 |
| :--- |


| [MBVE](https://arxiv.org/abs/1803.00101)\(Model-Based Value Expansion\): Feinberg et al, 2018 |
| :--- |


| [AlphaZero](https://arxiv.org/abs/1712.01815): Silver et al, 2017 |
| :--- |




