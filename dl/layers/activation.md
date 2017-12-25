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



激活函数的作用：

1，激活函数是用来加入非线性因素，解决模型所不能解决的问题。

2，激活函数可以用来组合训练数据的特征，特征的充分组合。

下面我分别对激活函数的两个作用进行解释。

  


1

加入非线性因素，解决非线性问题

  


  


![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68AlCm7A0UEAibQ78NC6sC35atnGlbTLxZy7IGACaR6vjwA1icwujwnukzg/640?tp=webp&wxfrom=5&wx_lazy=1)

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68AvxbYAYwI3Qt5Nw9Oo5rVT78vic4WNztcpFSZnZUxPDicmxd2gPIPdPkQ/640?tp=webp&wxfrom=5&wx_lazy=1)

好吧，很容易能够看出，我给出的样本点根本不是线性可分的，一个感知器无论得到的直线怎么动，都不可能完全正确的将三角形与圆形区分出来，那么我们很容易想到用多个感知器来进行组合，以便获得更大的分类问题，好的，下面我们上图，看是否可行

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68Aa9kjSQK3BaPaRyXYZ4lj9zw1n68Bzo0DOYHoia7wA7uNwNicyLa7Tdaw/640?tp=webp&wxfrom=5&wx_lazy=1)

好的，我们已经得到了多感知器分类器了，那么它的分类能力是否强大到能将非线性数据点正确分类开呢~我们来分析一下：

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68APxicIvDK8RBmQKXYwhXC3STICAYXSqYHIaick0U7xYK9JjWM5f4DibSaw/640?tp=webp&wxfrom=5&wx_lazy=1)

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68AqXl6EtxQGchoN1WpGU9c7ZWYjJNAbhXzTSAVaxbS9vPqdlFwgNXArg/640?tp=webp&wxfrom=5&wx_lazy=1)

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68AaF7vuo4QZM9rGIiaTYVQzEHGLUt4UhFaqFIicxv7DBuaVMvb57fHka6w/640?tp=webp&wxfrom=5&wx_lazy=1)

如果我们的每一个结点加入了阶跃函数作为激活函数的话，就是上图描述的

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68AIVofwh2Vf8VVicfQYOaXTn7RBF4dx8l4iaZ9XoMJicjJpuxpPzDazAgqQ/640?tp=webp&wxfrom=5&wx_lazy=1)

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68A91ibwaV5jndFHDQmLZSF8dibVCQaic3F0M5jR9jqf7YHX9145r9m4UZlg/640?tp=webp&wxfrom=5&wx_lazy=1)

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68A4YboW9tdLDiaC3xZEUYEsP7W2UehgfwZulk6WEV8SSolicgvscibZqeYw/640?tp=webp&wxfrom=5&wx_lazy=1)

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68A7BrWb5T70wic8of5Q9yp4F3gw2Ric56dQvO9vX7iaWW6iahXaNwbrtdRIg/640?tp=webp&wxfrom=5&wx_lazy=1)

那么随着不断训练优化，我们也就能够解决非线性的问题了~

所以到这里为止，我们就解释了这个观点，加入激活函数是用来加入非线性因素的，解决线性模型所不能解决的问题。

  


_下面我来讲解另一个作用_

  


2

 激活函数可以用来组合训练数据的特征，特征的充分组合

  


![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68AcaIwMnjibt4BO99UQL1icib2Hl7h70xMdOStY8iciaCmVoD1pMp9UGMPnKQ/640?tp=webp&wxfrom=5&wx_lazy=1)  


![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68AS4rMwvg6HEJPnJGyGziaRTicXcHXtEyGR8ryfcs42qiaeFvpz2FgwJxaQ/640?tp=webp&wxfrom=5&wx_lazy=1)

我们可以通过上图可以看出，立方激活函数已经将输入的特征进行相互组合了。

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68A9oN35byHd8clVAGHBHT4TQ1EFJqLV0ZXic84O6RrS5L3UOmEHsHqKwg/640?tp=webp&wxfrom=5&wx_lazy=1)

![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68Alibs6pOnlzyKkQBwRWkyItwaOlVe1Em6ekuZblQDouKQPqJ9iamG7iaNA/640?tp=webp&wxfrom=5&wx_lazy=1)

通过泰勒展开，我们可以看到，我们已经构造出立方激活函数的形式了。

于是我们可以总结如下：

  


3

总结

  


  


![](https://mmbiz.qpic.cn/mmbiz_png/nJZZib3qIQW7HVtOa5QfJy6tF1EKGj68Af2LJ8MSBKK70mXVoaj8ZjQovR5iasLwzESBppvNcfjFMBeL1hsya5fA/?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1)

这就把原来需要领域知识的专家对特征进行组合的情况，在激活函数运算后，其实也能够起到特征组合的作用。（只要激活函数中有能够泰勒展开的函数，就可能起到特征组合的作用）

这也许能给我们一些思考。

