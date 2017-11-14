## LocallyConnected1D层 {#locallyconnected1d}

---

`LocallyConnected1D`层与`Conv1D`工作方式类似，唯一的区别是不进行权值共享。即施加在不同输入位置的滤波器是不一样的。



## LocallyConnected2D层 {#locallyconnected2d}

---

`LocallyConnected2D`层与`Convolution2D`工作方式类似，唯一的区别是不进行权值共享。即施加在不同输入patch的滤波器是不一样的，当使用该层作为模型首层时，需要提供参数`input_dim`或`input_shape`参数。参数含义参考`Convolution2D`。





