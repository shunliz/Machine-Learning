RNN的思想是利用序列信息。在经典的神经网络中，我们认为所有的输入和输出是相互独立的。 但是对于许多其他任务这个想法是很不好的。如果你需要预测一个句子中的下一个字，知道前边的字会是很有帮助的。理论上RNN可以利用任意长度的序列的信息，但是实际只能处理很有限的前几步信息。下边是一个典型的RNN图示：

![](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/rnn.jpg)

上图是一个展开的RNN，也就是说如果我们要处理一个含有5个字的序列，这个网络就要展开成5层的神经网络，一层处理一个字。图中的各个符号解释如下：

$$x_t$$是第t步的输入，$$x_1$$可能是一个one-hot编码的的字。

$$s_t$$是第t步的隐藏状态，是网络的记忆功能。$$s_t$$是通过前边的隐藏状态和当前输入步计算得出：$$st=f(Uxt+Ws_{t-1})$$, f通常是tanh或者ReLU

$$o_t$$是第t步的输出，$$ot=softmax(Vs_t)$$





RNN在很多NLP任务中表现的非常出色。

### RNN通用来生成文本

通过给定一段文本，通过一定的训练，可以生成相似的文本。

### 机器翻译

![](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/Screen-Shot-2015-09-17-at-10.39.06-AM.png)

### 语音识别

### 生成文本描述

![](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/Screen-Shot-2015-09-17-at-11.44.24-AM-1024x349.png)



