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

GPTQ是在Optimal Brain Quantization基础上优化的，我们先了解一下OBQ。

#### Optimal Brain Quantization

我们先从我们要解决的问题开始。对于神经网络的每一层$l$, 我们需要找一个参数$\mathbf{W}_{\ell}$的量化版的$\widetilde{\mathbf{W}}_{\ell}$，使得模型的性能损失最小化，我们需要$\left(\widetilde{\mathbf{W}}_{\ell} \mathbf{X}_{\ell}\right)$的输出和$\left(\mathbf{W}_{\ell} \mathbf{X}_{\ell}\right)$的输出尽量接近，换句话说我们需要找到
$$
\arg \min _{\widetilde{\mathbf{W}}_{\ell}}\left\|\mathbf{W}_{\ell} \mathbf{X}_{\ell}-\widetilde{\mathbf{W}}_{\ell} \mathbf{X}_{\ell}\right\|_2^2
$$
有很多方法解决这个问题，这里我们介绍OBQ。OBQ是一个剪枝技术，从一个全连接的网络中删除一些权重。它使用一种近似的技术，选择最合适的单个权重参数$w_q$删除，同时优化更新$\delta_F$，调整没有量化的权重F来补偿这个删除操作。
$$
\begin{aligned}
& w_q=\arg \min _{w_q} \frac{\left(\operatorname{quant}\left(w_q\right)-w_q\right)^2}{\left[\mathbf{H}_F^{-1}\right]_{q q}}, \\
& \delta_F=-\frac{w_q-\operatorname{quant}\left(w_q\right)}{\left[\mathbf{H}_F^{-1}\right]_{q q}} \cdot\left(\mathbf{H}_F^{-1}\right)_{:, q} .
\end{aligned}
$$
其中quant(*w*)是量化的权重，$H_F$是hessian矩阵。

这个计算量是非常的大，虽然OBQ通过hessian矩阵和高斯消元技术减少计算量，但是计算量仍然很大。GPTQ做了重大改进是的OBQ可以使用在LLM中。

#### GPTQ

GPTQ 将权重分组（如：128列为一组）为多个子矩阵（block）。对某个 block 内的所有参数逐个量化，每个参数量化后，需要适当调整这个 block 内其他未量化的参数，以弥补量化造成的精度损失。因此，GPTQ 量化需要准备校准数据集。

GPTQ 量化过程如下图所示。首先，使用 Cholesky 分解求解 Hessian 矩阵的逆，然后在给定的步骤中对连续列的块（粗体）进行量化，并在该步骤结束时更新剩余的权重（蓝色）。量化过程在每个块内递归应用，白色中间列表示当前正在被量化。

![image-20250727213951571](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727213951571.png)

GPTQ 的创新点如下：

- 取消贪心算法：OBS 采用贪心策略，先量化对目标影响最小的参数；但 GPTQ 发现直接按顺序做参数量化，对精度影响也不大。这项改进使得参数矩阵每一行的量化可以做并行的矩阵计算（这意味着我们可以独立地对每一行执行量化。即所谓的 per-channel quantization）。对于大模型场景，这项改进使得量化速度快了一个数量级；
- Lazy Batch-Updates：OBQ 对权重一个个进行单独更新，作者发现性能瓶颈实际在于GPU的内存带宽，而且同一个特征矩阵W不同列间的权重更新是不会互相影响的。因此作者提出了延迟批处理的方法，通过延迟一部分参数的更新，一次处理多个（如：128）列，来缓解带宽的压力，大幅提升了计算速度。
- Cholesky(乔莱斯基) 分解：用 Cholesky 分解(一种分解矩阵的方法)求海森矩阵的逆，提前计算好所有需要的信息，在增强数值稳定性的同时，后续更新的过程中再计算，进一步减少了计算量。

GPTQ的伪代码如下所示。

![image-20250727214530579](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250727214530579.png)

#### 使用autoGPTQ量化LLM

```
!BUILD_CUDA_EXT=0 pip install -q auto-gptq transformers
import random

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import torch
from transformers import AutoTokenizer

# Define base model and output directory
model_id = "gpt2"
out_dir = model_id + "-GPTQ"

# Load quantize config, model and tokenizer
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=False,
)
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quantize with GPTQ
model.quantize(
    examples_ids,
    batch_size=1,
    use_triton=True,
)

# Save model and tokenizer
model.save_quantized(out_dir, use_safetensors=True)
tokenizer.save_pretrained(out_dir)

#load quantized model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Reload model and tokenizer
model = AutoGPTQForCausalLM.from_quantized(
    out_dir,
    device=device,
    use_triton=True,
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(out_dir)

#use quantized model to inference
from transformers import pipeline

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
result = generator("I have a dream", do_sample=True, max_length=50)[0]['generated_text']
print(result)
```

### SmoothQuant

SmoothQuant （论文：**SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**）是一种同时确保准确率且推理高效的训练后量化 (PTQ) 方法，可实现 8 比特权重、8 比特激活 (W8A8) 量化。由于权重很容易量化，而激活则较难量化，因此，SmoothQuant 引入平滑因子s来平滑激活异常值，通过数学上等效的变换将量化难度从激活转移到权重上。

![image.png](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7De13bef4e1e114061bf5066cf92284bb4%7Etplv-k3u1fbpfcp-jj-mark%3A3024%3A0%3A0%3A0%3Aq75.awebp)

常规的矩阵乘如下：

$Y=XW$

SmoothQuant 对激活进行 smooth，按通道除以 smoothing factor。为了保持线性层数学上的等价性，以相反的方式对权重进行对应调整。SmoothQuant 的矩阵乘如下：

$Y=(Xdiag(s)^{-1}) \cdot (diag(s)W)=\hat{X} \hat{W}$

$X \in R ^{T \times C_i} $ 在 channel 维度（列）上每个元素除以 $s_i$ ，$W \in R ^{ C_i \times C_0}$则在每行上每个元素乘以 $s_i$ 。这样 Y 在数学上是完全相等的，平滑因子s的计算公式下面会讲述。

**将量化难度从激活迁移到权重**

为了减小量化误差，可以为所有 channels 增加有效量化 bits。当所有 channels 都拥有相同的最大值时，有效量化 bits 将会最大。

一种做法是让 $s_j=max(|X_j|),j=1,2,...,C_j$ ， $C_j$ 代表第 j 个 input channel。各 channel 通过除以 $s_j$ 后，激活 channels 都将有相同的最大值，这时激活比较容易量化。但是这种做法会把激活的量化难度全部转向权重，导致一个比较大的精度损失。

另一种做法是让$s_j=1/max(|W_j|)$，这样权重 channels 都将有相同的最大值，权重易量化，但激活量化误差会很大。

因此，我们需要在 weight 和 activation 中平衡量化难度，让彼此均容易被量化。本文作者通过加入一个超参 $\alpha$ （迁移强度），来控制从激活值迁移多少难度到权重值。一个合适的迁移强度值能够让权重和激活都易于量化。$\alpha$ 太大，权重难以量化，$\alpha$ 太小激活难以量化。

平滑因子 s 是在校准样本上获得的，整个转换是离线执行的。在运行时，激活是平滑的，无需缩放，具体步骤如下：

1. 校准阶段（离线）

![image.png](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7D8812962b317b460081353eab706ea424%7Etplv-k3u1fbpfcp-jj-mark%3A3024%3A0%3A0%3A0%3Aq75.awebp)

平滑因子s的计算公式如下：

![image.png](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dd177f2df50bd4a1aa42bbf1868089c06%7Etplv-k3u1fbpfcp-jj-mark%3A3024%3A0%3A0%3A0%3Aq75.awebp)

- $\alpha$ 表示迁移强度，为一个超参数，控制将多少激活值的量化难度迁移到权重量化。
- $C$ 表示激活的输入通道数。

1. 平滑阶段（离线）

![image.png](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Daeb17a26f18748169aebc0e40d5ef53f%7Etplv-k3u1fbpfcp-jj-mark%3A3024%3A0%3A0%3A0%3Aq75.awebp)

$\hat{X}, \hat{W}$ 的计算公式如下：

- $\hat{W}=diag(s) W $
- $\hat{X}=X diag(s)^{-1}$

1. 推理阶段（在线，部署模型）

![image.png](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7D5f424660e5b24cb19d39453bf198afa9%7Etplv-k3u1fbpfcp-jj-mark%3A3024%3A0%3A0%3A0%3Aq75.awebp)

平滑之后的激活的计算公式如下：

$Y = \hat{X} \hat{W}$



### AffineQuant

这是一种追求极致准确性的高级量化技术。传统量化方法通常只做简单的缩放，而 AffineQuant 则通过引入一个可学习的"仿射变换矩阵"，在量化前对权重和激活值的分布进行优化，使其更"适合"被量化。这能最大程度地减少量化过程中引入的误差，从而在基准测试中取得了顶尖的性能表现。



## 量化感知的训练QAT

### QLORA

QLoRA是模型量化(Quantilization) 与LoRA的结合，通过量化降低模型参数的精度 (4 bit) 来降低显存消耗。

QLoRA并不是将大模型量化成4 bit然后优化这些4 bit的参数，也不是加一个4 bit的LoRA。QLoRA的逻辑是，在加载时将大模型量化成4 bit，但在某一部分的计算时，将该部分的4 bit的参数反量化 (dequant) 成16 bit进行运算，而加入的LoRA同样是16 bit的。所以，QLoRA优化的是非使用状态中的参数的存储，而计算所涉及的到的输入输出、模型参数均使用高精度，模型的梯度、优化器状态等的显存消耗与普通LoRA并无二致。

QLoRA有几个核心设计：

1. **[4-bit NormalFloat Quantization](https://zhida.zhihu.com/search?content_id=241240214&content_type=Article&match_order=1&q=4-bit+NormalFloat+Quantization&zhida_source=entity).** QLoRA基于模型参数值通常呈现正态分布的假设，设计了一种适用于该分布的量化策略。模型参数从高精度量化到低精度时，参数值会落到4 bit 对应的16个区间中，这些区间并不是均匀划分的，而是根据正态分布的概率密度划分的，尽可能使得落到每个区间的参数数量相等，这样才能最大化地利用4 bit 精度的表达能力。假如均匀划分，由于参数是零均值正态分布，大部分参数都会落入靠近0的少数区间，而变得不可分辨，使得 4 bit 取值空间的利用率较低。

2.  **[Double Quantization](https://zhida.zhihu.com/search?content_id=241240214&content_type=Article&match_order=1&q=Double+Quantization&zhida_source=entity).** 量化过程中，每组被量化的参数会被记录一个absmax（绝对值最大值），用于在计算时将参数反量化回原始值。这个absmax是高精度保存的，所以也会占用相当大的显存。因此，QLoRA采用与参数量化相似的策略，将这些absmax值也量化成4 bit，进一步降低显存占用。

   ![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-dc71f33dfeeec05877a4bc17de70ff57_1440w.jpg)

3. **Paged Optimizers.** 优化器分页，类似CPU内存的分页策略，当GPU紧张时，将一部分gradient checkpoint转移到CPU内存。

### AWQ (Activation-aware Weight Quantization)

AWQ 是一种智能的量化方法。它认为并非所有权重都同等重要。通过分析模型的"激活值"，AWQ 能识别出对模型性能至关重要的那一小部分权重（约 1%），并在量化过程中对它们进行特殊保护，不降低其精度。而其他 99% 的权重则被压缩到低位宽（如 4-bit）。这种方法通用性强，且能在推理时带来显著的（超过 3 倍）加速。

在AWQ中会出现两个缩放因子，区分下，**scale**表示用来提高重要权重的精度的缩放因子，**q_scale**表示量化权重时高精度数据类型到低精度数据类型的映射关系。

AWQ的核心假设是极少量模型权重（<3%）对模型精度有非常大的影响，找到这些模型权重并保持为高精度，有利于保证整个量化后模型的精度。在核心假设上有两个需要解决的问题，一个是如何找到这些高价值模型权重，一个是保持这些高价值模型权重精度的方法。

**注意**：在AWQ的量化原理图示中W都是按通道(列）量化。

#### 谁是重要的权重

对第一个问题，作为矩阵乘法来看比如X*W，输入矩阵X中较大的值对应的W中的权重对模型精度有更大影响。

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-9d04f418f902f6d6f87891f4a0078e3f_1440w.jpg)

使用矩阵输入特征作为评估权重重要性的依据

#### 如何保存权重

对于第二个问题，直接在一个权重矩阵中同时存放不同精度的权重值会让矩阵乘法变得异常复杂。采用同时缩放激活值和权重的方法来提高重要权重的精度，同时对权重矩阵的元素执行相同的量化。

作为对比的量化后计算公式：$$
Q(w) \cdot x=\Delta \cdot \operatorname{Round}\left(\frac{w s}{\Delta}\right) \cdot x
$$

AWQ量化后的计算公式：$$
Q(w \cdot s) \cdot \frac{x}{s}=\Delta^{\prime} \cdot \operatorname{Round}\left(\frac{w s}{\Delta^{\prime}}\right) \cdot x \cdot \frac{1}{s}
$$，这里s表示缩放系数

分析AWQ量化方法和没有增加缩放的误差，Round带来是舍入误差大家都一样。当s不大的时候对$$
\frac{\Delta^{\prime}}{\Delta}
$$ 影响小，随着s增大AWQ量化后误差缩小。
$$
\operatorname{Err}(Q(w) \cdot x) / \operatorname{Err}\left(Q(w \cdot s) \cdot \frac{x}{s}\right)=\frac{\Delta^{\prime}}{\Delta} * \frac{1}{s}
$$
公式中其他符号解释

量化函数： $$
\operatorname{Round}\left(\frac{w}{\Delta^{\prime}}\right)
$$

量化因子q_scale：![image-20250728102802613](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250728102802613.png) 

量化的位数：N ，如果是 4bit 量化，则N=4 ；如果是 8bit 量化，则 N=8

#### 搜索合适的scale

如何寻找合适的scale，方法是在一个scale的空间中寻找能使被量化模块量化后误差最小的scale。

实际是个优化问题，目标函数设定为

![image-20250728102922746](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250728102922746.png)

对于scale的空间做了简化，首先对输入张量求均值得到Sx，以Sx 作为基线。，$s=s_X^{\alpha}, \alpha \in [0,1]$ 构造搜索空间。通过改动$\alpha$ 的值，调整权重矩阵所有的scale值。

![img](https://pic1.zhimg.com/v2-15a9b0dd7e6a32211ceda5307120d944_1440w.jpg)

a)整个权值量化 b)保留对激活值影响大权值为高精度 c)量化前先放大消除单独保留高精度造成对硬件不优化的混合精度计算

## 动态量化技术

### BitsandBytes动态量化

BitsandBytes动态量化是一种在模型推理阶段对激活值进行动态量化的技术，其核心思想是根据每次输入数据的实时分布动态调整量化参数，以实现对激活值的有效量化。

**动态量化原理：**BitsandBytes动态量化通过实时监测激活值的分布范围，动态确定量化区间和量化步长。例如，在每次推理时，它会根据当前激活值的最大值和最小值，将激活值的范围划分为若干个等间距的区间，每个区间用一个整数来表示。这种方法能够灵活适应不同的输入数据分布，确保量化后的激活值能够较好地反映原始激活值的变化。
**性能优化：**动态量化技术在减少存储和计算需求的同时，能够较好地保持模型性能。实验表明，使用BitsandBytes动态量化后，模型的推理速度可以提升约2倍，而准确率下降不超过3%。这种性能优化效果使得动态量化技术在实际应用中具有较高的实用性。
**适用场景：**BitsandBytes动态量化特别适用于输入数据分布变化较大的场景。例如，在图像识别任务中，不同图像的特征分布可能存在较大差异，动态量化能够根据每张图像的特征动态调整量化参数，从而更好地保持模型性能。此外，在自然语言处理任务中，对于不同主题或风格的文本输入，动态量化也能够有效地适应激活值的变化。
**计算开销：**尽管动态量化能够灵活适应输入数据的变化，但其计算开销相对较大。由于每次推理都需要动态计算量化参数，这会增加一定的计算时间。与静态量化相比，动态量化的推理时间会增加约10%。然而，这种计算开销在实际应用中是可以接受的，因为它能够带来更好的性能表现。
与其他技术的结合：BitsandBytes动态量化可以与其他量化技术结合使用，以进一步优化模型性能。例如，它可以与权重量化技术结合，同时对权重和激活值进行量化，从而更全面地减少模型的存储和计算需求。此外，动态量化还可以与模型压缩技术结合，如剪枝和低秩分解，进一步提高模型的效率和性能。



更多模型量化参考https://www.zhihu.com/column/c_1258047709686231040
