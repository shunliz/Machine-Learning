**PPO: OpenAI 提出的一种解决 Policy Gradient 不好确定 Learning rate \(或者 Step size\) 的问题. 因为如果 step size 过大, 学出来的 Policy 会一直乱动, 不会收敛, 但如果 Step Size 太小, 对于完成训练, 我们会等到绝望. PPO 利用 New Policy 和 Old Policy 的比例, 限制了 New Policy 的更新幅度, 让 Policy Gradient 对稍微大点的 Step size 不那么敏感.**

![](/assets/reinforcemnt-opeai-ppo.png)



