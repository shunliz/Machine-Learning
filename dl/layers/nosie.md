## GaussianNoise层 {#gaussiannoise}

---

为数据施加0均值，标准差为`stddev`的加性高斯噪声。该层在克服过拟合时比较有用，你可以将它看作是随机的数据提升。高斯噪声是需要对输入数据进行破坏时的自然选择。

因为这是一个起正则化作用的层，该层只在训练时才有效。



## GaussianDropout层 {#gaussiandropout}

---

为层的输入施加以1为均值，标准差为`sqrt(rate/(1-rate)`的乘性高斯噪声

因为这是一个起正则化作用的层，该层只在训练时才有效。



## AlphaDropout {#alphadropout}

---

对输入施加Alpha Dropout

Alpha Dropout是一种保持输入均值和方差不变的Dropout，该层的作用是即使在dropout时也保持数据的自规范性。 通过随机对负的饱和值进行激活，Alphe Drpout与selu激活函数配合较好。





