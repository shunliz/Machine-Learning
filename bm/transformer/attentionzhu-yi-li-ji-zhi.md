# Self-Attention（自注意力机制）

![](/assets/llm-transformer-attention1.png)

上图是论文中 Transformer 的内部结构图，左侧为 Encoder block，右侧为 Decoder block。红色圈中的部分为**Multi-Head Attention**，是由多个**Self-Attention**组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention \(其中有一个用到 Masked\)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 \(Residual Connection\) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。

因为**Self-Attention**是 Transformer 的重点，所以我们重点关注 Multi-Head Attention 以及 Self-Attention，首先详细了解一下 Self-Attention 的内部逻辑。

## 3.1 Self-Attention 结构

![](/assets/llm-transformer-attention2.png)

上图是 Self-Attention 的结构，在计算的时候需要用到矩阵**Q\(查询\),K\(键值\),V\(值\)**。在实际中，Self-Attention 接收的是输入\(单词的表示向量x组成的矩阵X\) 或者上一个 Encoder block 的输出。而**Q,K,V**正是通过 Self-Attention 的输入进行线性变换得到的。

## 3.2 Q, K, V 的计算

Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵**WQ,WK,WV**计算得到**Q,K,V**。计算如下图所示，**注意 X, Q, K, V 的每一行都表示一个单词。**

![](/assets/llm-transformer-attention3.png)

## 3.3 Self-Attention 的输出

得到矩阵 Q, K, V之后就可以计算出 Self-Attention 的输出了，计算的公式如下：

![](/assets/llm-transformer-attention4.png)

公式中计算矩阵**Q**和**K**每一行向量的内积，为了防止内积过大，因此除以$$d_k$$的平方根。**Q**乘以**K**的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为**Q**乘以$$K^T$$，1234 表示的是句子中的单词。

![](/assets/llm-transformer-attention5.png)

得到 Softmax 矩阵之后可以和**V**相乘，得到最终的输出**Z**。

![](/assets/llm-transformer-attention6.png)

上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出$$Z_1$$

等于所有单词 i 的值$$V_i$$根据 attention 系数的比例加在一起得到，如下图所示：

![](/assets/llm-transformer-attention7.png)

## 3.4 Multi-Head Attention

在上一步，我们已经知道怎么通过 Self-Attention 计算得到输出矩阵 Z，而 Multi-Head Attention 是由多个 Self-Attention 组合形成的，下图是论文中 Multi-Head Attention 的结构图。

![](/assets/llm-transformer-attention8.png)





