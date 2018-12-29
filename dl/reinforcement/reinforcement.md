除了agent和环境之外，强化学习的要素还包括**策略\(Policy\)**、**奖励\(reward signal\)**、**值函数\(value function\)**、**环境模型\(model\)**，下面对这几种要素进行说明：

1. **策略\(Policy\)**
   ，策略就是一个从当环境状态到行为的映射；$$$$   

$$a_t = \mu_\theta(s_t)$$   确定型

$$a_t \sim \pi_\theta(\cdot|s_t)$$ 随机型

随机型又分为 **categorical policies 和diagonal Gaussian policies**

**categorical policies 通常用在离散动作空间的场景**

采样阶段，随机生成每一个动作的概率。

Log-Likelihood：计算每一个动作的概率，$$log\pi_\theta(a|s) = log[P_\theta(s)]_a$$

**diagonal Gaussian policies 通常用在连续动作空间的场景**

采样阶段，生成随机动作的概率 $$a = \mu_\theta(s) +\delta_\theta(s)\odot z$$    $$z\sim N(0,I)$$

Log-Likelihood: $$log\pi_\theta(a|s) = -\frac{1}{2}( \sum_{i=1}^{k}(\frac{(a_i-\mu_i)^2)}{\delta_i^2}))+klog2\pi)$$

下一步的表示：

$$s_{t+1} = f(s_t,a_t)$$

$$s_{t+1} \sim P(\odot|s_t, a_t)$$

1. **奖励\(reward signal\)**
   ，奖励是agent执行一次行为获得的反馈，强化学习系统的目标是最大化累积的奖励，在不同状态下执行同一个行为可能会得到不同的奖励；
2. **值函数\(value function\)**
   ，一种状态的value为从当前状态出发到停机状态所获得的累积的奖励；
3. ![](/assets/reinforcementlearing3.png)
4. **环境模型\(model\)**
   ，agent能够根据环境模型预测环境的行为，采用环境模型的强化学习方法称为基于模型\(model-based\)的方法，不采用环境模型的强化学习方法称为model-free方法。
5. ![](/assets/reinforcementlearning2.png)

![](/assets/reinforcementlearning1.png)

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




[^1]: Enter footnote here.

