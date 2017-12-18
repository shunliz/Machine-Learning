**PPO: OpenAI 提出的一种解决 Policy Gradient 不好确定 Learning rate \(或者 Step size\) 的问题. 因为如果 step size 过大, 学出来的 Policy 会一直乱动, 不会收敛, 但如果 Step Size 太小, 对于完成训练, 我们会等到绝望. PPO 利用 New Policy 和 Old Policy 的比例, 限制了 New Policy 的更新幅度, 让 Policy Gradient 对稍微大点的 Step size 不那么敏感.**

![](/assets/reinforcemnt-opeai-ppo.png)

![](/assets/reinforcement-deepmind-ppo.png)

总的来说 PPO 是一套 Actor-Critic 结构, Actor 想**最大化**`J_PPO`, Critic 想**最小化**`L_BL`.Critic 的 loss 好说, 就是减小 TD error. 而 Actor 的就是在 old Policy 上根据 Advantage \(TD error\) 修改 new Policy, advantage 大的时候, 修改幅度大, 让 new Policy 更可能发生. 而且他们附加了一个 KL Penalty \(惩罚项, 不懂的同学搜一下 KL divergence\), 简单来说, 如果 new Policy 和 old Policy 差太多, 那 KL divergence 也越大, 我们不希望 new Policy 比 old Policy 差太多, 如果会差太多, 就相当于用了一个大的 Learning rate, 这样是不好的, 难收敛.



DPPO 算法

使用多个线程 \(workers\) 平行在不同的环境中收集数据,运行PPO





