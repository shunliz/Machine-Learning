# 模型量化

## **模型量化简介**

现在的大模型参数动辄几十上百亿的参数，这么大的参数量，没有强大的GPU集群很难运行。 模型量化是一种用于减少神经网络模型大小和计算量的技术，将模型参数（如：权重）从高精度数据类型（如：float32）转换为低精度数据类型（如：int8 或 fp4）。模型量化通过以更少的位数表示数据，可以减少模型尺寸，进而减少在推理时的内存消耗，并且在一些低精度运算较快的处理器上可以增加推理速度，同时仍然可以保持模型的性能。。其核心原理是将模型中的浮点数参数（如32位浮点数）转换为低位宽的数值表示（如8位整数），从而在不显著降低模型精度的前提下，大幅减少模型的存储空间和计算资源消耗。

![图片](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7D65dc21f5119c6f48d855ebfdf32971a5.png)

其主要作用包括：

**降低存储需求：**大模型通常包含数十亿甚至数千亿个参数，以浮点数形式存储会占用大量内存。量化技术可以将浮点数参数转换为低位宽的整数，显著减少模型存储空间。例如，将32位浮点数量化为8位整数，存储空间可减少4倍。
**加速计算：**在硬件上，整数运算通常比浮点数运算更快。量化后的模型在推理时可以利用硬件对整数运算的优化，从而提高计算速度。例如，一些专用的AI芯片对8位整数运算有专门的加速单元，能够使模型推理速度提升数倍。
**提高能效比：**减少存储和计算需求意味着模型在运行时消耗的能源也会降低，这对于在移动设备或边缘设备上部署大模型尤为重要，可以延长设备的电池续航时间。

![image-20250727171447170](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727171447170.png)

### 模型量化的粒度

- per-tensor（又名 per-layer）量化：每层或每个张量只有一个缩放因子，张量内的所有值都被这个缩放因子量化。
- per-channel 量化：卷积核的每个通道都有不同的缩放因子。
- per-token 量化：针对激活而言，针对每一行进行量化。在LLM中，通常与per-channel 量化搭配使用，如：逐Token量化激活，逐通道量化权重。
- per-group/group-wise：，以组为单位。正如 **Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT** 中所说的那样，分组量化的一个特殊情况是，将每个密集矩阵视为一组，每个矩阵都可以有自己的量化范围。而更普遍的情况是**将每个密集矩阵按输出神经元进行分割，每个连续的 N 输出神经元作为一个组**。比如：GPTQ、AWQ中使用128个元素为一组进行量化。有些地方也称为子通道分组（Sub-channel-wise）量化，即将通道划分为更小的子组，以实现更细粒度的精度控制。

下图展示了不同的量化粒度；其中，d为模型大小/隐藏状态维度；h是一个MHSA（多头自注意）中的Head数。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7D9cde867673aca38b0d502bed04953678.png)

下面展示了一些量化方法中不同量化对象的量化粒度：

![image-20250727174924867](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727174924867.png)

### 模型量化对象

**模型量化对象**

- Weight：权重量化是最常见的量化对象。量化权重可达到减少模型内存占用空间。权重在训练完后固定，数值范围与输入无关，可离线完成量化，通常相对容易量化；
- Activation：实际上激活往往是占内存使用的大头，因此量化激活不仅可以大大减少内存占用。更重要的是，结合权重量化可以充分利用整数计算获得模型推理性能的提升。但激活输出随输入变化而变化，需要统计数据动态范围，通常更难量化。
- KV Cache：除了权重和激活量化之外，在大语言模型中的 KV 缓存也会消耗不少的内存。 因此，量化 KV 缓存对于提高模型长序列生成的吞吐量至关重要。
- Gradient：相对上面的量化对象，略微小众一些，主要用于训练场景。在训练深度学习模型时，梯度通常是浮点数，量化梯度可以在分布式计算中减少通信开销，同时，也可以减少反向传播时的开销。

 **量化技术的分类**
大模型量化技术主要分为以下几类：

* **权重量化：**对模型的权重参数进行量化。常见的量化方法包括：
  * **均匀量化：**将权重的取值范围均匀划分为若干个区间，每个区间用一个代表值来表示。例如，将权重的范围[-1, 1]划分为256个区间，每个区间用区间中点的8位整数来表示。这种方法简单直观，但可能在某些情况下无法很好地捕捉权重的分布特性。
  * **非均匀量化：**根据权重的分布特性进行量化，例如采用K-means聚类算法对权重进行聚类，每个聚类中心用一个量化值表示。这种方法可以更好地适应权重的分布，但量化过程相对复杂。
* **激活量化：**对模型的激活值（即中间层的输出）进行量化。激活量化通常在模型推理时进行，目的是减少激活值的存储和计算开销。常见的激活量化方法包括：

  * **动态量化**：在每次推理时根据激活值的实时分布动态确定量化参数。这种方法可以适应不同的输入数据，但量化过程需要在每次推理时进行，增加了计算开销。
  * **静态量化：**在模型训练完成后，对激活值进行一次性的量化，确定固定的量化参数。这种方法在推理时不需要动态计算量化参数，计算效率较高，但可能无法很好地适应不同的输入数据。
* **混合量化**：结合权重量化和激活量化，同时对模型的权重和激活值进行量化。混合量化可以更全面地减少模型的存储和计算需求，但量化过程更加复杂，需要在权重量化和激活量化之间进行协调和优化。
* **结构化量化**：除了对权重和激活值的数值进行量化外，还对模型的结构进行量化，例如将权重矩阵中的某些行或列置为零，或者将权重矩阵分解为稀疏矩阵和低秩矩阵的乘积。结构化量化可以在不显著降低模型性能的情况下，进一步减少模型的存储和计算需求，但可能会对模型的训练和优化过程带来更大的挑战。

**静态量化与动态量化**

通常，**对于激活而言**，静态量化是指如果采用具有代表性的校准数据集来为其生成缩放因子和零点，这些参数在模型的整个生命周期中保持不变。静态量化的优点在于推理时的计算效率较高，因为它不需要在运行时动态计算量化参数。然而，由于量化参数是固定的，静态量化可能会引入一些量化误差，从而影响模型的精度

而动态量化是指在每次前向传递期间计算激活的最小值和最大值，以提供动态的缩放因子以实现高精度。动态量化的优点在于它可以更准确地表示模型的激活值，因为它考虑了运行时的实际数据分布。然而，这种方法的缺点是可能会增加计算开销，因为需要在运行时计算量化参数。动态量化适合于那些对模型精度要求较高的应用场景，尤其是当模型的输入数据分布变化较大时。

目前，常见的是对激活使用静态量化，其中最小/最大范围是在离线校准阶段计算的。但由于LLM中激活范围差异巨大，将导致准确度显著下降。

**离线量化与在线量化**

离线量化是指模型上线前进行量化并生成缩放因子，而在线量化是指模型运行时进行量化。

动态与静态量化的区别在于是否使用校准集，而离线与在线量化的区别则是量化的时机不同。简单理解就是说**离线静态量化**是指在模型上线推理前使用校准集生成缩放因子，对权重和激活进行量化。**在线动态量化**是指在模型上线推理时，在每次前向传播过程中实时生成缩放因子，对模型对权重和激活进行量化。 而**离线动态量化**通常是指对权重在运行前先进行量化，对激活在运行时进行动态量化。

**线性量化与非线性量化**

根据量化数据表示的原始数据范围是否均匀，还可以将量化方法分为线性量化和非线性量化。实际的深度神经网络的权重和激活值通常是不均匀的；因此，理论上使用非线性量化导致的精度损失更小，但在实际推理中非线性量化的计算复杂度较高，通常使用线性量化。 下面着重介绍线性量化的原理。假设 表示量化前的浮点数，量化后的整数 可以表示为：

![image-20250727180151303](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727180151303.png)

其中， round(.)和clip(.) 分别表示取整和截断操作，qmin和qmax是量化后的最小值和最大值。

s为缩放系数，表示数据量化的间隔，其求解方式为![image-20250727180336935](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727180336935.png) 分别表示输入浮点数据中的最大值和最小值，qmax 、qmin分别表示量化后最大定点值和最小定点值。 z是表示数据偏移的偏置。 z为 0 的量化被称为对称量化， z不为 0 的量化称为非对称量化。对称量化可以避免量化算子在推理中计算z相关的部分，降低推理时的计算复杂度；非对称量化可以根据实际数据的分布确定最小值和最小值，可以更加充分的利用量化数据信息，使得量化导致的损失更低。

### 量化数据类型

LLM主要有三种类型量化：

- 仅权重量化：只量化每个线性层的权重张量W。
- 权重激活量化：量化每个线性层的输入激活X和权重张量W。
- KV缓存量化：量化每个自注意力块中的键张量K和值张量V。

下面列举了业界的一些量化数据类型的典型方案。

针对仅权重量化：

- 对于 W8A16 量化，代表方法有 MinMax
- 对于 W6A16 量化，代表方法有 FP6-LLM
- 对于 W4A16 量化，代表方法有 AWQ、GPTQ、SpQR、OmniQuant、QuIP#
- 对于 W3A16 量化，代表方法有 GPTQ、SpQR、OmniQuant、QuIP#
- 对于 W2A16 量化，代表方法有 OmniQuant、QuIP、QuIP#

针对权重激活量化：

- 对于 W8A8 量化，代表方法有 LLM.int8()、SmoothQuant、ZeroQuant
- 对于 W6A6 量化，代表方法有 OmniQuant
- 对于 W4A8 量化，代表方法有 QoQ
- 对于 W4A4 量化，代表方法有 Atom 、QuaRot、OmniQuant

针对 KV Cache量化：

- KV8：INT8（LMDeploy、TensorRT-LLM）、FP8（TensorRT-LLM、vLLM）
- KV4：Atom、QuaRot、QoQ
- KV3：KVQuant
- KV2：KVQuant、KIVI

**各种模型量化技术对比**

|       量化技术       |     类型     |                         精度损失控制                         |                     性能优化                     |                        硬件支持要求                        |                          适用场景                          |
| :------------------: | :----------: | :----------------------------------------------------------: | :----------------------------------------------: | :--------------------------------------------------------: | :--------------------------------------------------------: |
|         GPTQ         |  训练后量化  | 准确率下降不超过2%，保持较高性能，适合多种模型架构和任务场景 |     显著减少存储空间和计算需求，适合快速部署     | 对硬件的实时计算能力有一定要求，适合具有较强计算能力的硬件 |        资源受限的边缘设备、对模型性能要求适中的场景        |
|     SmoothQuant      |  训练后量化  | 通过优化激活值量化过程，显著降低量化误差，准确率下降不超过2% |         优化激活值量化精度，提升模型性能         |      对硬件的存储和计算能力要求适中，适合多种硬件环境      |    对激活值量化精度要求较高的任务，如情感分析、问答系统    |
|        QLoRA         | 量化感知训练 |   准确率下降不超过3%，通过低秩分解和量化集成提升模型适应性   |             将模型推理速度提升约2倍              |      对硬件的存储和计算能力要求适中，适合多种硬件环境      |     具有大规模权重矩阵的模型，对推理速度要求较高的场景     |
|         AWQ          | 量化感知训练 | 准确率下降不超过2%，通过激活值感知的权重量化显著降低量化误差 |             将模型推理速度提升约3倍              |     对硬件的计算能力和存储容量要求较高，适合高性能硬件     |       对模型性能要求较高的场景，如机器翻译、文本生成       |
| BitsandBytes动态量化 |   动态量化   |   准确率下降不超过3%，通过动态适应输入数据分布优化模型性能   | 推理速度略有增加，适合输入数据分布变化较大的场景 | 对硬件的实时计算能力有一定要求，适合具有较强计算能力的硬件 | 输入数据分布变化较大的场景，如图像识别、不同主题的文本输入 |

**模型量化可以分为两大类：**

**训练后量化Post-Training Quantization (PTQ)** ：模型训练完后，对参数进行量化。

**量化感知的训练Quantization-Aware Training (QAT)**：在pre-training和fine-tune阶段，对参数进行量化。

**核心技术原理**

![image-20250727171808934](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727171808934.png)

## 基本量化方法

### **零点量化（Zero-point quantization）**

零点量化通过线性映射，将原始数据的最小值和最大值对应到目标数据类型的最小值和最大值。例如，当我们将 FP16 数值转换为 Int8 时，就会用到这种量化方法。下图展示了这一过程。

![图片](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7De44ba688409f28f44ee7547691f776cd.png)

我们希望将数据缩放到 Int8 范围（在本例中为 0 ~ 255）。因此，我们需要计算数据在原始尺度中的相对值，并通过乘以 Int8 量化范围（255）来进行重新缩放。其计算公式如下所示：

![image-20250727181506986](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727181506986.png)

![image-20250727181528594](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727181528594.png)

其中，n 是用于量化的位数（在本例中为 8）。需要注意的是，如果我们希望将数值缩放到 -127 ~ 127，则需要从上述结果中减去 127，然后对结果进行舍入。

```python
def zeropoint_quantize(X):
    # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # Calculate scale
    scale = 255 / x_range

    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()

    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    # Dequantize
    X_dequant = (X_quant - zeropoint) / scale

    return X_quant.to(torch.int8), X_dequant
```

这种方法比其他量化方法能够更精确地表示原始数据。但另一方面，它需要更复杂的计算，因此在实际应用中，我们需要在精度和计算复杂度之间进行权衡。

### **绝对最大值量化（Absolute maximum quantization）**

绝对最大值量化（Absolute Maximum Quantization）将原始数据中的最大绝对值映射到目标数据类型的有符号范围。仍然以 FP16 到 Int8 的转换为例。为了简化问题，我们使用受限的量化范围，即 -127 ~ 127（供参考，完整的量化范围是 -128 ~ 127）。下图展示了这种量化方法。
![图片](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7D1c380d9042f30a6e8c4708a60b1afe2d.png)

我们首先计算数据中的**最大绝对值**，然后利用该值对原始数据进行重新缩放。其计算公式如下所示：

![image-20250727181145067](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727181145067.png)

```python
import torch
def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))
    # Quantize
    X_quant = (scale * X).round()
    # Dequantize
    X_dequant = X_quant / scale
    return X_quant.to(torch.int8), X_dequant
```

这些方法都属于 “舍入至最近值”（Round-to-Nearest, RTN） 量化技术。

然而，朴素量化（naive quantization） 存在精度下降的问题，因为减少位数的同时，也会丢失部分信息。因此，现代量化技术的目标是在减少位宽的同时尽可能降低精度损失。

### 量化GPT-2示例

```python
!pip install -q bitsandbytes>=0.39.0
!pip install -q git+https://github.com/huggingface/accelerate.git
!pip install -q git+https://github.com/huggingface/transformers.git

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

# Set device to CPU for now
device = 'cpu'

# Load model and tokenizer
model_id = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print model size
print(f"Model size: {model.get_memory_footprint():,} bytes")

# Extract weights of the first layer
weights = model.transformer.h[0].attn.c_attn.weight.data
print("Original weights:")
print(weights)

# Quantize layer using absmax quantization
weights_abs_quant, _ = absmax_quantize(weights)
print("\nAbsmax quantized weights:")
print(weights_abs_quant)

# Quantize layer using absmax quantization
weights_zp_quant, _ = zeropoint_quantize(weights)
print("\nZero-point quantized weights:")
print(weights_zp_quant)
```

目前，主要有两种主流的现代量化技术：

## 训练后量化PTQ

现代量化策略：为高效部署而生

量化不仅可以用于训练，更广泛的应用是在模型训练完成后，为了部署和推理进行压缩。这个过程被称为"训练后量化"（Post-Training Quantization, PTQ）。以下是几种前沿的 PTQ 策略。

### GPTQ



### AWQ (Activation-aware Weight Quantization)

AWQ 是一种智能的量化方法。它认为并非所有权重都同等重要。通过分析模型的"激活值"，AWQ 能识别出对模型性能至关重要的那一小部分权重（约 1%），并在量化过程中对它们进行特殊保护，不降低其精度。而其他 99% 的权重则被压缩到低位宽（如 4-bit）。这种方法通用性强，且能在推理时带来显著的（超过 3 倍）加速。

### AffineQuant

这是一种追求极致准确性的高级量化技术。传统量化方法通常只做简单的缩放，而 AffineQuant 则通过引入一个可学习的"仿射变换矩阵"，在量化前对权重和激活值的分布进行优化，使其更"适合"被量化。这能最大程度地减少量化过程中引入的误差，从而在基准测试中取得了顶尖的性能表现。



## 量化感知的训练QAT

### QLORA



## Unsloth

Unsloth 是一个流行的 LLM 训练优化库，它也提供了一套强大的动态量化方案。与静态量化不同，它能在运行时根据情况灵活调整精度，实现了高压缩率和高精度的完美平衡。例如，它可以将一个模型从 20GB 压缩到 6.5GB，同时保持极高的准确率。

虽然 Unsloth 不是量化方法，但我想介绍这个用于 LLM 的开源超高效微调库。Unsloth 是一个完全集成参数高效微调方法（如 LoRA 和 QLoRA）的库。它针对开源著名 LLM 优化了每个内核，并减少了高达约 80% 的内存使用率！例如，它支持DeepSeek-R1、Llama、Mistral、Phi、Qwen 和 Gemma 这些开源 LLM。
