随时间反向传播（BPTT）算法

---

先简单回顾一下RNN的基本公式：

$$s_t = \tanh (Ux_t+Ws_{t-1})$$

$$\hat y_t=softmax(Vs_t)$$

RNN的损失函数定义为交叉熵损失：

$$E_t(y_t,\hat y_t)=-y_t\log\hat y_t $$

$$E(y,\hat y)=\sum_{t}E_t(y_t, \hat y_t)=-\sum_{t}y_t\log\hat y_t$$

$$y_t$$是时刻t的样本实际值， $$\hat y\_t$$是预测值，我们通常把整个序列作为一个训练样本，所以总的误差就是每一步的误差的加和。![](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/rnn-bptt1.png)

