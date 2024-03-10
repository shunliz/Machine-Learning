# Transformer

首先介绍 Transformer 的整体结构，下图是 Transformer 用于中英文翻译的整体结构：

![](/assets/llm-transformer1.png)

可以看到**Transformer 由 Encoder 和 Decoder 两个部分组成**，Encoder 和 Decoder 都包含 6 个 block。Transformer 的工作流程大体如下：

**第一步：**获取输入句子的每一个单词的表示向量**X**，**X**由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到。

![](/assets/llm-transformer12.png)

**第二步：**

将得到的单词表示向量矩阵 \(如上图所示，每一行是一个单词的表示**x**\) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵**C**，如下图。单词向量矩阵用$$X_{n \times d}$$表示， n 是句子中单词个数，d 是表示向量的维度 \(论文中 d=512\)。每一个 Encoder block 输出的矩阵维度与输入完全一致。![](/assets/llm-transformer13.png)

**第三步**：将 Encoder 输出的编码信息矩阵**C**传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1，如下图所示。在使用的过程中，翻译到单词 i+1 的时候需要通过**Mask \(掩盖\)**操作遮盖住 i+1 之后的单词。

上图 Decoder 接收了 Encoder 的编码矩阵**C**，然后首先输入一个翻译开始符 "&lt;Begin&gt;"，预测第一个单词 "I"；然后输入翻译开始符 "&lt;

Begin&gt;" 和单词 "I"，预测单词 "have"，以此类推。这是 Transformer 使用时候的大致流程，接下来是里面各个部分的细节。

