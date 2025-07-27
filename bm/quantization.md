# 模型量化

现在的大模型参数动辄几十上百亿的参数，这么大的参数量，没有强大的GPU集群很难运行。 所以有了量化技术，在不修改模型的情况下，通过降低模型参数的精度，减小加载模型参数的需要的内容空间。

模型量化可以分为两大类：

**训练后量化Post-Training Quantization (PTQ)** ：模型训练完后，对参数进行量化。

**量化感知的训练Quantization-Aware Training (QAT)**：在pre-training和fine-tune阶段，对参数进行量化。

## 训练后量化PTQ



## 量化感知的训练QAT

