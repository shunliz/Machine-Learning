# Transformer 的输入

Transformer 中单词的输入表示**x**由**单词 Embedding**和**位置 Embedding**（Positional Encoding）相加得到。

![](/assets/llm-transformer-input1.png)

### 2.1 单词 Embedding {#h_338817680_3}

单词的 Embedding 有很多种方式可以获取，例如可以采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到。

### 2.2 位置 Embedding {#h_338817680_4}

Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要。**所以 Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。

位置 Embedding 用**PE**表示，**PE**的维度与单词 Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：![](/assets/llm-transformer-input2.png)

其中，pos 表示单词在句子中的位置，d 表示 PE的维度 \(与词 Embedding 一样\)，2i 表示偶数的维度，2i+1 表示奇数维度 \(即 2i≤d, 2i+1≤d\)。使用这种公式计算 PE 有以下的好处：

* 使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
* 可以让模型容易地计算出相对位置，对于固定长度的间距 k，**PE\(pos+k\)**可以用**PE\(pos\)**
  计算得到。因为 Sin\(A+B\) = Sin\(A\)Cos\(B\) + Cos\(A\)Sin\(B\), Cos\(A+B\) = Cos\(A\)Cos\(B\) - Sin\(A\)Sin\(B\)。

将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量**x**，**x**就是 Transformer 的输入。

