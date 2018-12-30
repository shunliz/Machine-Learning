# 1. BLEU

> BLEU \(Bilingual Evaluation Understudy\) is an algorithm for evaluating the quality of text which has been **machine-translated **from one **natural language **to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU. BLEU was one of the first metrics to achieve a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metric. -- 维基百科

解释一下，首先bleu是一种**文本评估算法**，它是用来评估**机器翻译**跟**专业人工翻译**之间的对应关系，核心思想就是**机器翻译越接近专业人工翻译，质量就越好**，经过bleu算法得出的分数可以作为机器翻译质量的其中一个指标。还有另外两种METEOR和NIST评价指标。

## 为什么要用BLEU？

现实中很多时候我们需要用人工来评价翻译结果的，但这种方式非常慢，并且成本非常高，因为你需要请足够专业的翻译人员才能给出相对靠谱的翻译评估结果，一般这种人工评价都偏主观，并且非常依赖专业水平和经验。

为了解决这一问题，机器翻译领域的研究人员就发明了一些自动评价指标比如BLEU，METEOR和NIST等，在这些自动评价指标当中，**BLEU是目前最接近人类评分**的。



