## LeakyReLU层 {#leakyrelu}

---

LeakyRelU是修正线性单元（Rectified Linear Unit，ReLU）的特殊版本，当不激活时，LeakyReLU仍然会有非零输出值，从而获得一个小梯度，避免ReLU可能出现的神经元“死亡”现象。即，`f(x)=alpha * x for x <0 f(x) = x for x>=0`



## PReLU层 {#prelu}

---

该层为参数化的ReLU（Parametric ReLU），表达式是：`f(x) = alpha * x for x <0`,`f(x) = x for x>=0`，此处的`alpha`

为一个与xshape相同的可学习的参数向量。



## ELU层 {#elu}

---

ELU层是指数线性单元（Exponential Linera Unit），表达式为： 该层为参数化的ReLU（Parametric ReLU），表达式是：

`f(x) = alpha * (exp(x) - 1.) for x < 0`,`f(x) = x for x>=0`



## ThresholdedReLU层 {#thresholdedrelu}

---

该层是带有门限的ReLU，表达式是：`f(x) = x for x >theta`,`f(x) = 0 otherwise`

