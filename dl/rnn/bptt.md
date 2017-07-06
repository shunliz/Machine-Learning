随时间反向传播（BPTT）算法

---

先简单回顾一下RNN的基本公式：

$$s_t = \tanh (Ux_t+Ws_{t-1})$$

$$\hat y_t=softmax(Vs_t)$$

RNN的损失函数定义为交叉熵损失：

$$E_t(y_t,\hat y_t)=-y_t\log\hat y_t $$

$$E(y,\hat y)=\sum_{t}E_t(y_t, \hat y_t)=-\sum_{t}y_t\log\hat y_t$$

$$y_t$$是时刻t的样本实际值， $$\hat y\_t$$是预测值，我们通常把整个序列作为一个训练样本，所以总的误差就是每一步的误差的加和。![](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/rnn-bptt1.png)我们的目标是计算损失函数的梯度，然后通过梯度下降方法学习出所有的参数U, V, W。比如：$$\frac{\partial E}{\partial W}=\sum_{t}\frac{\partial E_t}{\partial W}$$

![](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/rnn-bptt-with-gradients.png)It's the right idea, but the RNN update equations in the question are non-standard. Generally the hidden state is computed as a linear combination of the previous hidden state and the input at that time. I will go through and replace the equations, but I think that your formulation is not wrong:

> Just for the forward pass:
>
> **Forward Pass 1**
>
> a\_0 = x\_0 \* u\_0

Assuming that the input for timetisx\_tand that we are dealing with scalar inputs and parameters \(as opposed to vectors\)

> b\_0 = s\_{-1} \* w\_0

Assumingsis the state

> z\_0 = a\_0 \* b\_0

Generally,z\_0 = a\_0 + b\_0 + kfor some constant k. Intuitively, this is because the hidden state is then a combination of the input at timetand the hidden state from the previous timet-1. In this way, they both contribute to the hidden state, but not synergistically. For many applications of RNN, this seems to be a good assumption \(they work well\). I am not sure whether the optimization of a network that uses products of the previous hidden state with the current input would be stable in optimization.

> s\_0 = func\_0\(z\_0\)\(wherefunc\_0is sig, or tanh\)

s\_0 = func\(z\_0\)--- no need to index the activation function by time step.

Also, there is no real need to index the parametersu\_0, w\_0, although it does help because ultimately when we derive the gradient of the parameters, we sum over time steps. Really, though, in describing the forward pass you can just say

z\_0 = u\*x\_0 + w\*s\_{-1} + k

s\_0 = funct\(z\_0\)

or

z\_t = u\*x\_t + w\*s\_{t-1} + k

s\_t = func\(z\_t\)

> **Foward Pass 2**
>
> a\_1 = x\_1 \* u\_1
>
> b\_1 = s\_0 \* w\_1
>
> z\_1 = a\_1 \* b\_1
>
> s\_1 = func\_1\(z\_1\)\(wherefunc\_1is sig, or tanh\)
>
> q = s\_1 \* v\_1

z\_1 = u\*x\_1 + w\*s\_{0} + k

s\_1 = funct\(z\_1\)

q = s\_1\*v1+c

Note that what you've defined here is actually not a standard RNN, but a many-to-one RNN:

[![](https://i.stack.imgur.com/B15TJm.png "Many-to-one RNN")](https://i.stack.imgur.com/B15TJm.png)

For more detail on this and RNNs in general, I would definitely recommend[Goodfellow et al RNN chapter](http://www.deeplearningbook.org/contents/rnn.html)and Andrej Karpathy's[post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)and[minimal character RNN](https://gist.github.com/karpathy/d4dee566867f8291f086)implementation.

> **Output pass**
>
> o = func\_2\(q\)\(wherefunc\_2is softmax\)
>
> E = func\_3\(o\)\(wherefunc\_3is x-entropy\)

o = soft\(q\)\(wheresoftis softmax\)

E = L\(o\)\(whereLis x-entropy\)

> Now, attempting to hand-bomb back prop, for U \(by working backwards through the above network\).\partial E/\partial u = \partial E/\partial u\_1 + \partial E/\partial u\_0
>
> \partial E/\partial u\_1 = \partial E/do \* \partial o/\partial q \* \partial q/\partial s\_1 \* \partial s\_1/\partial z\_1 \* \partial z\_1/\partial a\_1 \* \partial a\_1/\partial u\_1
>
> \partial E/\partial u\_0 = \partial E/\partial o \* \partial o/\partial q \* \partial q/\partial s\_1 \* \partial s\_1/\partial z\_1 \* \partial z\_1/\partial b\_1 \* \partial b\_1/\partial s\_0 \* \partial s\_0/dz\_0 \* \partial z\_0/\partial a\_0 \* \partial a\_0/\partial u\_0
>
> **Gathering like terms**
>
> \partial E/\partial u = \partial E/\partial o \* \partial o/\partial q \* \partial q/\partial s\_1 \* \partial s\_1/\partial z\_1 \* \(\(\partial z\_1/\partial a\_1 \* \partial a\_1/\partial u\_1\) + \(\partial z\_1/\partial b\_1 \* \partial b\_1/\partial s\_0 \* \partial s\_0/\partial z\_0 \* \partial z\_0/\partial a\_0 \* \partial a\_0/\partial u\_0\)\)
>
> **Making substitutions**
>
> \partial E/\partial u = \partial E/\partial o \* \partial o/\partial q \* v\_1 \* \partial s\_1/\partial z\_1 \* \(\(1 \* x\_1\) + \(1 \* w\_1 \* \partial s\_0/\partial z\_0 \* 1 \* x\_0\)\)
>
> **Ending with a nice, clean formula.**
>
> \partial E/\partial u = \partial E/\partial o \* \partial o/\partial q \* v\_1 \* \partial s\_1/\partial z\_1 \* \(x\_1 + w\_1 \* \partial s\_0/\partial z\_0 \* x\_0\)

For u, the derivative is

\dfrac{\partial{L}}{\partial{u}}=\sum\_t \dfrac{\partial{L}}{\partial{u\_t}} = \dfrac{\partial L}{\partial o} \dfrac{\partial o}{\partial s\_1} \dfrac{\partial s\_1}{\partial u\_1}+\dfrac{\partial L}{\partial o} \dfrac{\partial o}{\partial s\_1}\dfrac{\partial s\_1}{\partial s\_0}\dfrac{\partial s\_0}{\partial u\_0}

> **And similarly**
>
> \partial E/\partial w = \partial E/\partial o \* \partial o/\partial q \* v\_1 \* \partial s\_1/\partial z\_1 \* \(s\_0 + w\_1 \* \partial s\_0/\partial z\_0 \* s\_{-1}\)

For w:

\dfrac{\partial{L}}{\partial{w}}=\sum\_t \dfrac{\partial{L}}{\partial{w\_t}} = \dfrac{\partial L}{\partial o} \dfrac{\partial o}{\partial s\_1} \dfrac{\partial s\_1}{\partial w\_1}+\dfrac{\partial L}{\partial o} \dfrac{\partial o}{\partial s\_1}\dfrac{\partial s\_1}{\partial s\_0}\dfrac{\partial s\_0}{\partial w\_0}

