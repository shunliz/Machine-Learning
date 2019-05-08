# 最简单的梯度策略推导

对于随机参数化的策略， 我们的目标是最大化期望回报：$$J\left(\pi_{\theta}\right)=\underset{\tau \sim \pi_{\theta}}{\mathrm{E}}[R(\tau)]$$。为了推导我们这里的$$R(\tau)$$是有限无加权的回报，有限有加权的推导是相同的。

我们可以通过梯度上升优化策略，如

$$\theta_{k+1}=\theta_{k}+\alpha \nabla_{\theta} J\left.\left(\pi_{\theta}\right)\right|_{\theta_{k}}$$

$$\nabla_{\theta} J\left(\pi_{\theta}\right)$$叫做梯度策略，这样优化策略的方法我们叫做梯度策略算法，如Vanilla梯度策略，TRPO， PPO。

先列几个我们推导会用到的公式

1， 策略轨迹的概率。假设策略来自$$\pi_\theta$$，策略轨迹$$\tau=\left(s_{0}, a_{0}, \dots, s_{T+1}\right)$$的概率表示为下

$$P(\tau | \theta)=\rho_{0}\left(s_{0}\right) \prod_{t=0}^{T} P\left(s_{t+1} | s_{t}, a_{t}\right) \pi_{\theta}\left(a_{t} | s_{t}\right)$$

2，Log求导的一些技巧

$$\nabla_{\theta} P(\tau | \theta)=P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta)$$ 用到了logx的导数是1/x和链式法则

3，策略轨迹的log概率，

$$\log P(\tau | \theta)=\log \rho_{0}\left(s_{0}\right)+\sum_{t=0}^{T}\left(\log P\left(s_{t+1} | s_{t}, a_{t}\right)+\log \pi_{\theta}\left(a_{t} | s_{t}\right)\right)$$

4，环境函数的梯度。环境和$$\theta$$无关，所以$$\rho_{0}\left(s_{0}\right), P\left(s_{t+1} | s_{t}, a_{t}\right)$$和$$R(\tau)$$是0.

5，$$\begin{aligned} \nabla_{\theta} \log P(\tau | \theta) &=\nabla_{\theta} \log \rho_{0}\left(s_{0}\right)+\sum_{t=0}^{T}\left(\nabla_{\theta} \log P\left(s_{t+1} | s_{t}, a_{t}\right)+\nabla_{\theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right)\right) \\ &=\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right) \end{aligned}$$

上边几步合到一块的推导过程

$$\begin{aligned} \nabla_{\theta} J\left(\pi_{\theta}\right) &=\nabla_{\theta} \underset{_{\tau} \sim \pi_{\theta}}{E}[R(\tau)] \\ &=\nabla_{\theta} \int_{\tau} P(\tau | \theta) R(\tau) \\ &=\int_{\tau} \nabla_{\theta} P(\tau | \theta) R(\tau) \\ &=\int_{\tau} P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta) R(\tau) \\ &=\underset{\tau \sim \pi_{\theta}}{E}\left[\nabla_{\theta} \log P(\tau | \theta) R(\tau)\right] \end{aligned}$$

$$\therefore \nabla_{\theta} J\left(\pi_{\theta}\right)=\underset{\tau \sim \pi_{\theta}}{E}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right) R(\tau)\right]$$

$$\hat{g}=\frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} | s_{t}\right) R(\tau)$$

