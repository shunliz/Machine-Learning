BERT 模型是 Google 在 2018 年提出的一种 NLP 模型，成为最近几年 NLP 领域最具有突破性的一项技术。在 11 个 NLP 领域的任务上都刷新了以往的记录，例如GLUE，SquAD1.1，MultiNLI 等。

**1. 前言**

Google 在论文《BERT: Pre-training of Deep Bidirectional Transformers forLanguage Understanding》中提出了 BERT 模型，BERT 模型主要利用了 Transformer 的 Encoder 结构，采用的是最原始的 Transformer，对 Transformer 不熟悉的童鞋可以参考一下之前的文章Transformer 模型详解或者 Jay Alammar 的博客:The Illustrated Transformer。总的来说 BERT 具有以下的特点：

**结构：**采用了 Transformer 的 Encoder 结构，但是模型结构比 Transformer 要深。Transformer Encoder 包含 6 个 Encoder block，BERT-base 模型包含 12 个 Encoder block，BERT-large 包含 24 个 Encoder block。

**训练：**训练主要分为两个阶段：预训练阶段和 Fine-tuning 阶段。预训练阶段与 Word2Vec，ELMo 等类似，是在大型数据集上根据一些预训练任务训练得到。Fine-tuning 阶段是后续用于一些下游任务的时候进行微调，例如文本分类，词性标注，问答系统等，BERT 无需调整结构就可以在不同的任务上进行微调。

**预训练任务1：**BERT 的第一个预训练任务是 **Masked LM**，在句子中随机遮盖一部分单词，然后同时利用上下文的信息预测遮盖的单词，这样可以更好地根据全文理解单词的意思。Masked LM 是 BERT 的重点，和 biLSTM 预测方法是有区别的，后续会讲到。

**预训练任务2：**BERT 的第二个预训练任务是 **Next Sentence Prediction (NSP)**，下一句预测任务，这个任务主要是让模型能够更好地理解句子间的关系。

**2. BERT 结构**

![img](../assets/F980CB1A8FE4491B4EC0ADC8030090B3)BERT 结构

上图是 BERT 的结构图，左侧的图表示了预训练的过程，右边的图是对于具体任务的微调过程。

**2.1 BERT 的输入**

BERT 的输入可以包含一个句子对 (句子 A 和句子 B)，也可以是单个句子。同时 BERT 增加了一些有特殊作用的标志位：

[CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 **C** 可以用于后续的分类任务。[SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。[MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么。例如给定两个句子 "my dog is cute" 和 "he likes palying" 作为输入样本，BERT 会转为 "[CLS] my dog is cute [SEP] he likes play ##ing [SEP]"。BERT 里面用了 WordPiece 方法，会将单词拆成子词单元 (SubWord)，所以有的词会拆出词根，例如 "palying" 会变成 "paly" + "##ing"。

BERT 得到要输入的句子后，要将句子的单词转成 Embedding，Embedding 用 **E**表示。与 Transformer 不同，BERT 的输入 Embedding 由三个部分相加得到：Token Embedding，Segment Embedding，Position Embedding。

![img](../assets/0DB4E913CDE0490352F049530300B030)BERT 的输入 Embedding

**Token Embedding：**单词的 Embedding，例如 [CLS] dog 等，通过训练学习得到。

**Segment Embedding：**用于区分每一个单词属于句子 A 还是句子 B，如果只输入一个句子就只使用 **E**A，通过训练学习得到。

**Position Embedding：**编码单词出现的位置，与 Transformer 使用固定的公式计算不同，BERT 的 Position Embedding 也是通过学习得到的，在 BERT 中，假设句子最长为 512。

**2.2 BERT 预训练**

BERT 输入句子中单词的 Embedding 之后，通过预训练方式训练模型，预训练有两个任务。

第一个是 Masked LM，在句子中随机用 [MASK] 替换一部分单词，然后将句子传入 BERT 中编码每一个单词的信息，最终用 [MASK] 的编码信息 **T**[MASK] 预测该位置的正确单词。

第二个是下一句预测，将句子 A 和 B 输入 BERT，预测 B 是否 A 的下一句，使用 [CLS] 的编码信息 **C**进行预测。

BERT 预训练的过程可以用下图来表示。

![img](../assets/1FA07D2383DE55C81EEC0DC20200E0B2)BERT 预训练过程

**2.3 BERT 用于具体 NLP 任务**

预训练得到的 BERT 模型可以在后续用于具体 NLP 任务的时候进行微调 (**Fine-tuning 阶段**)，BERT 模型可以适用于多种不同的 NLP 任务，如下图所示。

![img](../assets/F980CB1A590F40CC18DC34DB030050B1)BERT 用于不同任务

**一对句子的分类任务：**例如自然语言推断 (MNLI)，句子语义等价判断 (QQP) 等，如上图 (a) 所示，需要将两个句子传入 BERT，然后使用 [CLS] 的输出值 **C**进行句子对分类。

**单个句子分类任务：**例如句子情感分析 (SST-2)，判断句子语法是否可以接受 (CoLA) 等，如上图 (b) 所示，只需要输入一个句子，无需使用 [SEP] 标志，然后也是用 [CLS] 的输出值 **C**进行分类。

**问答任务：**如 SQuAD v1.1 数据集，样本是语句对 (Question, Paragraph)，Question 表示问题，Paragraph 是一段来自 Wikipedia 的文本，Paragraph 包含了问题的答案。而训练的目标是在 Paragraph 找出答案的起始位置 (Start，End)。如上图 (c) 所示，将 Question 和 Paragraph 传入 BERT，然后 BERT 根据 Paragraph 所有单词的输出预测 Start 和 End 的位置。

**单个句子标注任务：**例如命名实体识别 (NER)，输入单个句子，然后根据 BERT 对于每个单词的输出 **T**预测这个单词的类别，是属于 Person，Organization，Location，Miscellaneous 还是 Other (非命名实体)。

**3. 预训练任务**

预训练部分是 BERT 的重点，接下来了解 BERT 预训练的细节。BERT 包括两个预训练任务 **Masked LM**和 **下一句预测**。

**3.1 Masked LM**

我们先回顾一下以往语言模型的预训练方法，使用句子 "我/喜欢/学习/自然/语言/处理" 为例。在训练语言模型的时候通常需要进行一些 **Mask**操作，防止信息泄露问题，信息泄露指在预测单词 "自然" 的时候，提前得知 "自然" 的信息。后面会讲到 Transformer Encoder 信息泄露的原因。

**Word2Vec 的 CBOW：**通过单词 i 的上文和下文信息预测单词 i，但是采用的是词袋模型，不知道单词的顺序信息。例如预测单词 "自然" 的时候，会同时采用上文 "我/喜欢/学习" 和下文 "语言/处理" 进行预测。CBOW 在训练时是相当于把 "自然" 这个单词 Mask 的。

**ELMo：**ELMo 在训练的时候使用 biLSTM，预测 "自然" 的时候，前向 LSTM 会 Mask "自然" 之后的所有单词，使用上文 "我/喜欢/学习" 预测；后向 LSTM 会 Mask "自然" 之前的单词，使用下文 "语言/处理" 进行预测。然后再将前向 LSTM 和后向 LSTM 的输出拼接在一起，因此 ELMo 是将上下文信息分隔开进行预测的，而不是同时利用上下文信息进行预测。

**OpenAI GPT：**OpenAI GPT 是另外一种使用 Transformer 训练语言模型的算法，但是 OpenAI GPT 使用的是 Transformer 的 Decoder，是一种单向的结构。预测 "自然" 的时候只使用上文 "我/喜欢/学习"，Decoder 中包含了 Mask 操作，将当前预测词之后的单词都 Mask。

下图显示了 BERT 和 ELMo、OpenAI GPT 的区别。

![img](../assets/E898EA1AC809E00358CC44D2030080B3)BERT ELMo 和 OpenAI GPT

BERT 的作者认为在预测单词时，要同时利用单词 left (上文) 和 right (下文) 信息才能最好地预测。将 ELMo 这种分别进行 left-to-right 和 right-to-left 的模型称为 **shallow bidirectional model (浅层双向模型)**，BERT 希望在 Transformer Encoder 结构上训练出一种深度双向模型 **deep bidirectional model**，因此提出了 Mask LM 这种方法进行训练。

**Mask LM 是用于防止信息泄露的**，例如预测单词 "自然" 的时候，如果不把输入部分的 "自然" Mask 掉，则预测输出的地方是可以直接获得 "自然" 的信息。

![img](../assets/3FA07D238FE05C114E6508DA020030B3)BERT 的 Masked LM

BERT 在训练时只预测 [Mask] 位置的单词，这样就可以同时利用上下文信息。但是在后续使用的时候，句子中并不会出现 [Mask] 的单词，这样会影响模型的性能。因此在训练时采用如下策略，随机选择句子中 15% 的单词进行 Mask，在选择为 Mask 的单词中，有 80% 真的使用 [Mask] 进行替换，10% 不进行替换，剩下 10% 使用一个随机单词替换。

例如句子 "my dog is hairy"，选择了单词 "hairy" 进行 Mask，则：

80% 的概率，将句子 "my dog is hairy" 转换为句子 "my dog is [Mask]"。10% 的概率，保持句子为 "my dog is hairy" 不变。10% 的概率，将单词 "hairy" 替换成另一个随机词，例如 "apple"。将句子 "my dog is hairy" 转换为句子 "my dog is apple"。以上是 BERT 的第一个预训练任务 Masked LM。

**3.2 下一句预测**

BERT 的第二个预训练任务是 **Next Sentence Prediction (NSP)**，即下一句预测，给定两个句子 A 和 B，要预测句子 B 是否是句子 A 的下一个句子。

BERT 使用这一预训练任务的主要原因是，很多下游任务，例如问答系统 (QA)，自然语言推断 (NLI) 都需要模型能够理解两个句子之间的关系，但是通过训练语言模型达不到这个目的。

BERT 在进行训练的时候，有 50% 的概率会选择相连的两个句子 A B，有 50% 的概率会选择不相连得到两个句子 A B，然后通过 [CLS] 标志位的输出 **C**预测句子 A 的下一句是不是句子 B。

输入 = [CLS] 我 喜欢 玩 [Mask] 联盟 [SEP] 我 最 擅长 的 [Mask] 是 亚索 [SEP]类别 = B 是 A 的下一句输入 = [CLS] 我 喜欢 玩 [Mask] 联盟 [SEP] 今天 天气 很 [Mask] [SEP]类别 = B 不是 A 的下一句**4. BERT 总结**

因为 BERT 预训练时候采用了 Masked LM，每个 batch 只会训练 15% 的单词，因此需要更多的预训练步骤。ELMo 之类的顺序模型，会对每一个单词都进行预测。

BERT 使用了 Transformer 的 Encoder 和 Masked LM 预训练方法，因此可以进行双向预测；而 OpenAI GPT 使用了 Transformer 的 Decoder 结构，利用了 Decoder 中的 Mask，只能顺序预测。