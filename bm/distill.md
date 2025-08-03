# 模型蒸馏

模型蒸馏（Model Distillation）是一种经典的模型压缩与知识迁移技术，核心思想是让**小模型（学生模型，Student Model）** 学习**大模型（教师模型，Teacher Model）** 的 “知识”，从而在大幅减小模型规模、降低计算成本的同时，保持与大模型接近的性能。

## 基本概念

**教师模型**：通常是一个性能优异但参数量大、计算复杂的模型（如大参数量的 Transformer、ResNet 等），已通过训练在任务上达到较高精度。
**学生模型**：一个结构更简单、参数量更小、计算效率更高的模型（如轻量级 CNN、小尺寸 Transformer 等），目标是通过学习教师模型的 “知识”，在性能上接近教师模型。
**核心目标**：解决 “大模型性能好但部署难，小模型部署易但性能差” 的矛盾，实现 “小模型高效能”。

## 模型蒸馏原理


![image-20250803100329471](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250803100329471.png)
![image-20250803092143490](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250803092143490.png)

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7De61d1b15888fbb8f186ca7e067e43787.png)

![image-20250803100139747](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dimage-20250803100139747.png)

### **蒸馏的关键**

蒸馏的核心是定义 “知识” 的形式，并设计有效的学习方式。在分类任务中，最经典的 “知识” 是教师模型输出的软标签（Soft Labels），其获取依赖于温度参数（Temperature）。

1. **软标签：比硬标签更丰富的知识**
**硬标签（Hard Labels）**：原始数据的真实标签（如分类任务中的独热编码，“猫” 对应 [1,0,0]，“狗” 对应 [0,1,0]），仅包含 “正确类别” 的信息，忽略类间关系。
**软标签（Soft Labels）**：教师模型通过 “温度调节的 Softmax” 输出的概率分布，包含类间相似性信息（如 “猫” 与 “老虎” 更接近，“狗” 与 “狼” 更接近）。
例如，一张 “老虎” 的图片，教师模型的硬标签是 “老虎”（独热编码），但软标签可能显示 “老虎” 的概率为 80%，“猫” 为 15%，“狗” 为 5%—— 这种分布揭示了 “老虎与猫更相似” 的隐性知识，对学生模型学习更有价值。

2. **温度参数（Temperature, T）**
软标签通过在 Softmax 函数中引入温度参数生成。标准 Softmax 公式为： $$\delta(z)_i=\frac {e^{z_i}} {\sum_j e^{z_j}}$$

加入温度 T 后（称为 “蒸馏 Softmax”）： $$\delta_T(z)_i=\frac {\frac {e^{z_i}} T} {\sum_j \frac {e^{z_j}} T}$$
作用：控制概率分布的 “柔软度”。
T=1 时，与标准 Softmax 一致，输出接近硬标签（概率集中在少数类别）；
T 越大，概率分布越平缓（“越软”），类间差异的细节越明显（如 “老虎” 和 “猫” 的概率差缩小，更易体现相似性）

## 模型蒸馏实战

##### Step1. 准备

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
```

##### Step2. 定义Distiller模型

```python
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
```

##### Step3. 准备好teacher, student模型

```python
# Create the teacher
teacher = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="teacher",
)

# Create the student
student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="student",
)

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)
```

##### Step4. 准备好数据集

```python
# Prepare the train and test dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))

x_test = x_test.astype("float32") / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1))
```

##### Step5. 训练teacher模型

```python
# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate teacher on data. Teacher网络比较大, 需要更多轮次保证模型不会欠拟合
teacher.fit(x_train, y_train, epochs=6)
teacher.evaluate(x_test, y_test)
```

##### Step5. 蒸馏训练出一个student模型

```python
# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
distiller.fit(x_train, y_train, epochs=5)

# Evaluate student on test dataset
distiller.evaluate(x_test, y_test)
```

##### step6. 独立训练一个student模型用于对比

```python
# Train student as doen usually
student_scratch.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train and evaluate student trained from scratch.
student_scratch.fit(x_train, y_train, epochs=3)
student_scratch.evaluate(x_test, y_test)

def model_test(model):     
  num_images = 256
  start = time.perf_counter()
  for _ in range(num_images):
      index = random.randint(0, x_test.shape[0])
      x = x_test[index]
      y = y_test[index]
      x.shape = (1, 28, 28, 1)  # 变成[[]]
      predict = model.predict(x)
      predict = np.argmax(predict)  # 取最大值的位置

  end = time.perf_counter()
  time_ir = end - start

  print(
      f"model in Inference Engine/CPU: {time_ir/num_images:.4f} "
      f"seconds per image, FPS: {num_images/time_ir:.2f}"
  )

model_test(student_scratch)
```

##### step7. 模型性能测试

```python
def model_test(model):     
  num_images = 256
  start = time.perf_counter()
  for _ in range(num_images):
      index = random.randint(0, x_test.shape[0])
      x = x_test[index]
      y = y_test[index]
      x.shape = (1, 28, 28, 1)  # 变成[[]]
      predict = model.predict(x)
      predict = np.argmax(predict)  # 取最大值的位置

  end = time.perf_counter()
  time_ir = end - start

  print(
      f"model in Inference Engine/CPU: {time_ir/num_images:.4f} "
      f"seconds per image, FPS: {num_images/time_ir:.2f}"
  )
  
model_test(teacher)
model_test(distiller)
model_test(student_scratch)
```

##### Step8. 实验结果

| 模型名称        | 模型大小 | 模型评估                   | 模型性能(-n 200)                       |
| :-------------- | :------- | :------------------------- | :------------------------------------- |
| teacher model   | 5.46M    | loss: 0.0565 aux: 0.9855   | per image: : 0.0561seconds  FPS: 17.82 |
| distiller model | 0.80M    | loss: 0.0525 aux: 0.9801   | per image: : 0.0502seconds  FPS: 19.86 |
| student model   | 0.80M    | loss: 0.06131 auc: 0.97129 | per image: 0.0502 seconds  FPS: 19.92  |

## **蒸馏 vs 微调**

![img](https://raw.githubusercontent.com/shunliz/picbed/master/img/%24%7Byear%7D/%24%7Bmonth%7D/%24%7Bday%7D/%24%7Bfilename%7Dv2-bc7dc9b8055baaed54e4457f907d3160_1440w.jpg)

## 变体与扩展

随着研究发展，模型蒸馏已从分类任务扩展到更广泛的场景，衍生出多种变体：

**自蒸馏（Self-Distillation）**：无需单独教师模型，用模型自身的中间层输出或不同训练阶段的预测作为 “教师信号”（如让模型的早期 epoch 输出监督后期 epoch）。
**多教师蒸馏（Multi-Teacher Distillation）**：多个教师模型（不同结构或训练数据）共同监督学生，融合多样化知识以提升性能。
**跨任务蒸馏**：教师模型与学生模型任务不同（如教师做目标检测，学生做图像分类），通过提取教师的通用特征实现知识迁移。
**领域扩展**：从图像分类扩展到 NLP（如 BERT 蒸馏为 DistilBERT）、目标检测（如用 Faster R-CNN 蒸馏轻量级检测器）、语音识别等领域。

## 应用场景

**移动端 AI**：如手机拍照的实时美颜、场景识别（用蒸馏后的小模型减少功耗）；
**实时推理**：自动驾驶中的障碍物检测（需低延迟）、推荐系统中的实时排序（快速响应用户请求）；
**资源受限场景**：物联网设备（如智能家居传感器）的本地数据处理（算力有限，无法运行大模型）。