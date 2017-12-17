Google DeepMind 提出的一种使用`Actor Critic`结构, 但是输出的不是行为的概率, 而是具体的行为, 用于连续动作 \(continuous action\) 的预测.`DDPG`结合了之前获得成功的`DQN`结构, 提高了`Actor Critic`的稳定性和收敛性.



## 算法 {#算法}

`DDPG`的算法实际上就是一种`Actor Critic`,

[![](https://morvanzhou.github.io/static/results/reinforcement-learning/6-2-0.png "Deep Deterministic Policy Gradient \(DDPG\) \(Tensorflow\)-0")](https://morvanzhou.github.io/static/results/reinforcement-learning/6-2-0.png)

关于`Actor`部分, 他的参数更新同样会涉及到`Critic`, 上面是关于`Actor`参数的更新, 它的前半部分`grad[Q]`是从`Critic`来的, 这是在说:**这次`Actor`的动作要怎么移动, 才能获得更大的`Q`**, 而后半部分`grad[u]`是从`Actor`来的, 这是在说:**`Actor`要怎么样修改自身参数, 使得`Actor`更有可能做这个动作**. 所以两者合起来就是在说:**`Actor`要朝着更有可能获取大`Q`的方向修改动作参数了**.

[![](https://morvanzhou.github.io/static/results/reinforcement-learning/6-2-1.png "Deep Deterministic Policy Gradient \(DDPG\) \(Tensorflow\)-1")](https://morvanzhou.github.io/static/results/reinforcement-learning/6-2-1.png)

上面这个是关于`Critic`的更新, 它借鉴了`DQN`和`Double Q learning`的方式, 有两个计算`Q`的神经网络,`Q_target`中依据下一状态, 用`Actor`来选择动作, 而这时的`Actor`也是一个`Actor_target`\(有着 Actor 很久之前的参数\). 使用这种方法获得的`Q_target`能像`DQN`那样切断相关性, 提高收敛性.

