## Decoder 结构 {#h_338817680_14}

![](/assets/llm-transformer-decoder1.png)

上图红色部分为 Transformer 的 Decoder block 结构，与 Encoder block 相似，但是存在一些区别：

* 包含两个 Multi-Head Attention 层。
* 第一个 Multi-Head Attention 层采用了 Masked 操作。
* 第二个 Multi-Head Attention 层的**K, V**矩阵使用 Encoder 的**编码信息矩阵C**进行计算，而**Q**使用上一个 Decoder block 的输出计算。
* 最后有一个 Softmax 层计算下一个翻译单词的概率。

### 5.1 第一个 Multi-Head Attention {#h_338817680_15}

Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。下面以 "我有一只猫" 翻译成 "I have a cat" 为例，了解一下 Masked 操作。

下面的描述中使用了类似 Teacher Forcing 的概念，不熟悉 Teacher Forcing 的童鞋可以参考以下上一篇文章Seq2Seq 模型详解。在 Decoder 的时候，是需要根据之前的翻译，求解当前最有可能的翻译，如下图所示。首先根据输入 "&lt;Begin&gt;" 预测出第一个单词为 "I"，然后根据输入 "&lt;Begin&gt; I" 预测下一个单词 "have"。

![](/assets/llm-transformer-decoder2.png)

Decoder 可以在训练的过程中使用 Teacher Forcing 并且并行化训练，即将正确的单词序列 \(&lt;Begin&gt; I have a cat\) 和对应输出 \(I have a cat &lt;end&gt;\) 传递到 Decoder。那么在预测第 i 个输出时，就要将第 i+1 之后的单词掩盖住，**注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 0 1 2 3 4 5 分别表示 "&lt;Begin&gt; I have a cat &lt;end&gt;"。**

**第一步：**是 Decoder 的输入矩阵和**Mask**矩阵，输入矩阵包含 "&lt;Begin&gt; I have a cat" \(0, 1, 2, 3, 4\) 五个单词的表示向量，**Mask**是一个 5×5 的矩阵。在**Mask**可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。

![](/assets/llm-transforme-decoder5.png)

**第二步：**

接下来的操作和之前的 Self-Attention 一样，通过输入矩阵**X**计算得到**Q,K,V**矩阵。然后计算**Q**和$$K^T$$的乘积$$QK^T$$。

![](/assets/llm-transformer-decoder51.png)

**第三步：**

在得到$$QK^T$$之后需要进行 Softmax，计算 attention score，我们在 Softmax 之前需要使用**Mask**矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

![](/assets/llm-transformer-decoder53.png)

得到**Mask**$$QK^T$$之后在**Mask**$$QK^T$$上进行 Softmax，每一行的和都为 1。但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0。

**第四步：**使用**Mask**$$QK^T$$与矩阵**V**相乘，得到输出**Z**，则单词 1 的输出向量$$Z_1$$是只包含单词 1 信息的。

![](/assets/llm-transformer-decoder55.png)

**第五步：**

通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵$$Z_i$$，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出$$Z_i$$, 然后计算得到第一个 Multi-Head Attention 的输出**Z**，**Z**与输入**X**维度一样。

### 5.2 第二个 Multi-Head Attention {#h_338817680_16}

Decoder block 第二个 Multi-Head Attention 变化不大， 主要的区别在于其中 Self-Attention 的**K, V**矩阵不是使用 上一个 Decoder block 的输出计算的，而是使用**Encoder 的编码信息矩阵 C**计算的。

根据 Encoder 的输出**C**计算得到**K, V**，根据上一个 Decoder block 的输出**Z**计算**Q**\(如果是第一个 Decoder block 则使用输入矩阵**X**进行计算\)，后续的计算方法与之前描述的一致。

这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 \(这些信息无需**Mask**\)。

### 5.3 Softmax 预测输出单词 {#h_338817680_17}

Decoder block 最后的部分是利用 Softmax 预测下一个单词，在之前的网络层我们可以得到一个最终的输出 Z，因为 Mask 的存在，使得单词 0 的输出 Z0 只包含单词 0 的信息，如下：

![](/assets/llm-transformer-decoder61.png)

Softmax 根据输出矩阵的每一行预测下一个单词：

![](/assets/llm-transformer-decoder62.png)

## 6. Transformer 总结 {#h_338817680_18}

* Transformer 与 RNN 不同，可以比较好地并行训练。
* Transformer 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则 Transformer 就是一个词袋模型了。
* Transformer 的重点是 Self-Attention 结构，其中用到的**Q, K, V**矩阵通过输出进行线性变换得到。
* Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score。



