## Recurrent层 {#recurrent_1}

---

这是循环层的抽象类，请不要在模型中直接应用该层（因为它是抽象类，无法实例化任何对象）。请使用它的子类`LSTM`，`GRU`或`SimpleRNN`。

所有的循环层（`LSTM`,`GRU`,`SimpleRNN`）都继承本层，因此下面的参数可以在任何循环层中使用

## SimpleRNN层 {#simplernn}

---

全连接RNN网络，RNN的输出会被回馈到输入

## GRU层 {#gru}

---

门限循环单元

## LSTM层 {#lstm}

---

Keras长短期记忆模型

## ConvLSTM2D层 {#convlstm2d}

---

ConvLSTM2D是一个LSTM网络，但它的输入变换和循环变换是通过卷积实现的

## SimpleRNNCell层 {#simplernncell}

---

SinpleRNN的Cell类

## GRUCell层 {#grucell}

---

GRU的Cell类

## LSTMCell层 {#lstmcell}

---

LSTM的Cell类

## StackedRNNCells层 {#stackedrnncells}

---

这是一个wrapper，用于将多个recurrent cell包装起来，使其行为类型单个cell。该层用于实现搞笑的stacked RNN

## CuDNNGRU层 {#cudnngru}

---

基于CuDNN的快速GRU实现，只能在GPU上运行，只能使用tensoflow为后端

## CuDNNLSTM层 {#cudnnlstm}

---

基于CuDNN的快速LSTM实现，只能在GPU上运行，只能使用tensoflow为后端



