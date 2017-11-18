K分类，第k类的参数为$$\theta_k$$, 组成二维矩阵$$\theta_{k*n}$$

概率： $$p(c=k|x;\theta)=\frac {exp(\theta^T_kx)} {\sum _{l=1} ^K exp(\theta^T_l x)}$$, k=1,2,....K

似然 函数：![](/assets/softmax1.png)



对数似然：![](/assets/softmax2.png)



 随机梯度：![](/assets/softmax3.png)

