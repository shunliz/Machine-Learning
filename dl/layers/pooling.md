池化层通过对数据进行分区采样，把一个大的矩阵降采样成一个小的矩阵，减少计算量，同时可以防止过拟合。

![](/assets/deeplayer-pooling1.png)

通常有最大池化层，平均池化层。最大池化层对每一个小区域选最最大值作为池化结果，平均池化层选取平均值作为池化结果。

## MaxPooling1D层 {#maxpooling1d}

---

对时域1D信号进行最大值池化

## MaxPooling2D层 {#maxpooling2d}

---

为空域信号施加最大值池化

## MaxPooling3D层 {#maxpooling3d}

---

为3D信号（空域或时空域）施加最大值池化

本层目前只能在使用Theano为后端时可用

## AveragePooling1D层 {#averagepooling1d}

---

对时域1D信号进行平均值池化

## AveragePooling2D层 {#averagepooling2d}

---

为空域信号施加平均值池化

## AveragePooling3D层 {#averagepooling3d}

---

为3D信号（空域或时空域）施加平均值池化

本层目前只能在使用Theano为后端时可用

## GlobalMaxPooling1D层 {#globalmaxpooling1d}

---

对于时间信号的全局最大池化

## GlobalAveragePooling1D层 {#globalaveragepooling1d}

---

为时域信号施加全局平均值池化

## GlobalMaxPooling2D层 {#globalmaxpooling2d}

---

为空域信号施加全局最大值池化

## GlobalAveragePooling2D层 {#globalaveragepooling2d}

---

为空域信号施加全局平均值池化

