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

**确定策略**

```
obs = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
net = mlp(obs, hidden_dims=(64,64), activation=tf.tanh)
actions = tf.layers.dense(net, units=act_dim, activation=None)
```

**随机策略**

随机策略通常分为两大类：类别策略和斜角高斯策略。

类别策略通常用在离散动作空间，斜角高斯策略通常用在连续动作空间。

无论哪种策略，下边两种计算都是十分重要：

* 从策略中采样动作
* 计算动作之间的相似度, $$log\pi_\theta(a|s)$$



**trajectories**

一系列状态和动作集合：

$$\tau =(s_0,a_0,s_1,a_1,.....)$$



**奖励和回报**

奖励和回报对于强化学习十分重要，它由当前环境的状态，刚才采取的动作和环境的下一个状态来决定。

$$r_t=R(s_t,a_t,s_{t+1})$$



**强化学习问题**

选择一个策略来最大化期望回报。

一个T步的策略轨迹可以表示为：

$$P(\tau|\pi)=\rho_0(s_0)\prod_{t=0}^{T-1} P(s_{t+1}|s_t,a_t)\pi(a_t|s_t))$$

回报表示为：

$$J(\pi)=\int_{\tau}^{} P(\tau|\pi)R(\tau)=E_{T\sim \pi}[R(\tau)]$$

强化学习问题可以表示为：

$$\pi^*=arg max_\pi J(\pi)$$

$$\pi^*$$是优化策略。



**价值函数**

通常有4中价值函数：

1. on-policy价值函数，从状态s开始，一直使用策略$$\pi$$.    $$V^\pi(s)=E_{\tau\sim\pi}[R(\tau)|s_0=s]$$
2. on-policy 动作价值函数， 从状态s开始，采用一个动作a，然后之后开始按照策略执行动作。 $$Q^\pi(s,a)=E_{\tau\sim\pi}[R(\tau)|s_0=s,a_0=a]$$
3. 优化价值函数，从s开始一直使用优化的策略。 $$V^*(s)=max_\pi E_{\tau\sim\pi}[R(\tau)|s_0=s]$$
4. 优化动作价值函数，从s开始，采取动作a，之后一直采取最优化策略动作。 $$Q^*(s,a)=max_\pi E_{\tau\sim\pi}[R(\tau)|s_0=s,a_0=a]$$



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




