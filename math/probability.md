# 第二章 概率论和信息论 {#第三章-概率论和信息论}

这一章主要是讨论概率论和信息论相关的内容。 概率论是一种用来表示不确定状态的数学方法。在人工智能当中，概率论的应用主要在两个方面：

1. 概率论的laws告诉我们AI系统如何去完成推论，也就是用来设计AI的推论结构；

2. 可以采用概率论和统计理论来理论的分析所提出的人工智能系统的性能。 信息理论的作用就是让我们来量化一个概率分布当中的不确定性。

## 为什么采用概率论 {#为什么采用概率论}

在机器学习当中，算法必须总是处理不确定量以及随机量，同时硬件中的错误也会经常发生。不确定性的来源有三个：  
 1. 系统的固有不确定性被建模包含。例如，大多数的量子力学将亚原子的运动描述为不确定性的。  
 2. 不可完全观测。即使是确定性系统也会表现出随机性，因为我们肯能不能观察到所有的驱动这个系统的变量。在Monty Hall问题中，输出结果已经给定了，但是对于contestant来讲，输出却是不确定的，有点类似于薛定谔的猫。  
 3. 不完整的建模。例如在机器人中，为了估计机器人的位置从而离散空间，这样就没有办法精确的知道objects的位置。  
 在很多情况下，一个简单的不确定的系统比复杂的确定的系统更好用。需要指出的一点是，虽然我们想找到一种描述不确定的方法，但是概率论并没所有我们需要的所有工具。概率论最先是用来研究事物发生的频率的，通常是对于可重复事件，重复多次，就会发生那么多次，例如概率为p，可能会有p次发生。但是对于举出的医生的例子，认为flu的概率是40%，这里就是指的置信度的概念（degree of belief）。前一种被称作为频率论的概率论Frequentist probability，后一种用来衡量不确定性水平的被称作为贝叶斯概率论（Bayesian probability）。

## 随机事件与概率 {#随机变量}

公式名称                                                             公式表达

德摩公式                                                  $$\overline{A \cup B}=\overline{A} \cap \overline{B}, \overline{A \cap B}=\overline{A} \cup \overline{B}$$

古典概率                                                 $$P(A)=m/n$$ =  A包含的基本事件数/基本事件总数

求逆公式                                                 $$P(\overline{A})=1-P(A)$$

加法公式                                                $$P(A \cup B)=P(A)+P(B)-P(A B)$$  当 $$P(AB)=0$$时，$$P(A U B)=P(A)+P(B)$$

减法公式                                                $$P(A-B)=P(A)-P(A B), B \subset A$$时$$P(A-B)=P(A)-P(B)$$

条件概率公式                                         $$P(B | A)=\frac{P(A B)}{P(A)} \quad P(A B)=P(A) P(B | A)=P(B) P(A | B)$$  
$$P(A B C)=P(A) P(B | A) P(C | A B)$$  
全概率公式                                              $$P(A)=\sum_{i=1}^{n} P\left(B_{i}\right) P\left(A | B_{i}\right)$$

贝叶斯公式                                              $$P\left(B_{i} | A\right)=\frac{P\left(B_{i}\right) P\left(A | B_{i}\right)}{\sum_{i=1}^{n} P\left(B_{i}\right) P\left(A | B_{i}\right)}$$

乘法公式                                                  $$P\left(A_{1} A_{2}\right)=P\left(A_{1}\right) P\left(A_{2} | A_{1}\right)=P\left(A_{2}\right) P\left(A_{1} | A_{2}\right)$$  
                                                                $$P\left(A_{1} A_{2} \cdots A_{n}\right)=P\left(A_{1}\right) P\left(A_{2} | A_{1}\right) P\left(A_{3} | A_{1} A_{2}\right) \cdots P\left(A_{n} | A_{1} A_{2} \cdots A_{n-1}\right)$$

两个事件相互独立                                    $$P(A B)=P(A) P(B) ; \quad P(B | A)=P(B) ; \quad P(B | A)=P(B | \overline{A})$$

三个事件独立性

1. A,B,C两两独立$$\Leftrightarrow P(A B)=P(A) P(B);P(B C)=P(B) P(C) ; P(A C)=P(A) P(C)$$
2. A,B,C相互独立$$\Leftrightarrow P(A B)=P(A) P(B);P(B C)=P(B) P(C) ; P(A C)=P(A) P(C) ; P(A B C)=P(A) P(B) P(C)$$

## 随机变量及其分布

**1，分布函数**

$$F(x)=P(X \leq x)=\left\{\begin{array}{ll}{\sum_{x_{2} \leq x} P\left(X=x_{k}\right)} \\ {\int_{-\infty}^{x} f(t) d t}\end{array}\right. \quad P(a<X \leq b)=F(b)-F(a)$$

* $$0 \leq F(x) \leq 1$$
* F\(x\)单调不减
* 右连续$$F(x+0)=F(x)$$
* $$F(-\infty)=0, F(+\infty)=1$$

**2， 离散型随机变量及其分布**

0-1分布  $$\mathrm{X} \sim \mathrm{b}(1, \mathrm{p})$$                                               $$P(X=k)=p^{k}(1-p)^{1-k}, \quad k=0,1$$

二项分布 $$X \sim B\left(n_{s}, p\right) $$                                          $$P(X=k)=C_{n}^{k} p^{k}(1-p)^{n-k}, \quad k=0,1, \cdots, n$$

泊松分布  $$\mathrm{x} \sim \mathrm{p}(\lambda)$$                                                   $$P(X=k)=\frac{\lambda^{k}}{k !} e^{-\lambda}, \quad k=0,1,2, \cdots$$

**3，连续性随机变量及其分布**

分布名称                                    密度函数                                                                      分布函数

均匀分布 $$x \sim U(a, b)$$          $$f(x)=\left\{\begin{array}{ll}{\frac{1}{b-a},} & {a<x<b} \\ 0,   else \end{array}\right.$$          $$F(x)=\left\{\begin{array}{cc}{0,} & {x<a} \\ {\frac{x-a}{b-a}, a} & { \leq x<b} \\ {1,} & {x \geq b}\end{array}\right.$$               $$F(x)=\left\{\begin{array}{cc}{0,} & {x<a} \\ {\frac{x-a}{b-a},} & {a \leq x<b} \\ {1,} & {x \geq b}\end{array}\right.$$

指数分布 $$\mathrm{X} \sim \mathrm{E}(\lambda)$$             $$f(x)=\left\{\begin{array}{cc}{\lambda e^{-\lambda x},} & {x>0} \\ {0,} & {x \leq 0}\end{array}\right.$$                                   $$F(x)=\left\{\begin{array}{cc}{1-e^{-j x},} & {x>0} \\ {0,} & {x \leq 0}\end{array}\right.$$

正态分布 $$\mathrm{x} \sim \mathrm{N}\left(\mu, \sigma^{2}\right)$$      $$\begin{aligned} f(x)=& \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(x-\mu)^{2}}{2 c^{2}}} \\ &-\infty<x<+\infty \end{aligned}$$                                  $$F(x)=\frac{1}{\sqrt{2 \pi} \sigma} \int_{-\infty}^{x} e^{-\frac{(t-\mu)^{2}}{2 \sigma^{2}}} \mathrm{d} t$$

标准正太分布 $$\mathrm{x} \sim \mathrm{N}(0,1)$$     $$\begin{array}{r}{\varphi(x)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{x^{2}}{2}}} \\ {-\infty<x<+\infty}\end{array}$$                                                $$\Phi(x)=\frac{1}{\sqrt{2 \pi}} \int_{-\infty}^{x} e^{-\frac{1}{2} t^{2}} d t$$

几何分布  $$x \sim G(p)$$                 $$P(X=k)=(1-p)^{k-1} p, 0<p<1, k=1,2, \cdots$$

超几何分布: $$H(N, M, n) : P(X=k)=\frac{C_{M}^{k} C_{N-l l}^{n-k}}{C_{N}^{n}}, k=0,1, \cdots, \min (n, M)$$

**重要公式与结论**

* $$X \sim N(0,1) \Rightarrow \varphi(0)=\frac{1}{\sqrt{2 \pi}}, \Phi(0)=\frac{1}{2}, \Phi(-a)=P(X \leq-a)=1-\Phi(a)$$
* $$X \sim N\left(\mu, \sigma^{2}\right) \Rightarrow \frac{X-\mu}{\sigma} \sim N(0,1), P(X \leq a)=\Phi\left(\frac{a-\mu}{\sigma}\right)$$
* $$X \sim E(\lambda) \Rightarrow P(X>s+t | X>s)=P(X>t)$$
* $$X \sim G(p) \Rightarrow P(X=m+k | X>m)=P(X=k)$$
* 离散型随机变量的分布函数为阶梯间断函数；连续型随机变量的分布函数为连续函数，但不一定为处处可导函数

* 存在既非离散也非连续型随机变量

**分布函数**

连续性随机变量                      $$F(x)=P(X \leq x)=\int_{-\infty}^{x} f(t) d t$$

离散型随机变量                      $$F(x)=P(X \leq x)=\sum_{k \leq x} P(X=k)$$

分布函数与密度函数的重要关系       $$F^{\prime}(x)=f(x)$$

**4，随机变量函数**$$Y=g(X)$$**的分布**

$$f_{Y}(y)=f_{X}(h(y)) \cdot\left|h^{\prime}(y)\right|(x=h(y)$$ h\(y\)是g\(x\)的反函数

## 多维随机变量及其分布

1，离散型二维随机变量及其分布

分布律   $$P\left(X=x_{i}, Y=y_{j}\right)=p_{i j}, i, j=1,2, \cdots$$              分布函数 $$F(X, Y)=\sum_{x_i\leq x} \sum_{y_j \leq y} p_{ij}$$

边缘分布   $$p_{i}=P\left(X=x_{i}\right)=\sum_{j} p_{i j} \quad p_{j}=P\left(Y=y_{j}\right)=\sum_{i} p_{i j}$$

条件分布律  $$P\left(X=x_{i} | Y=y_{j}\right)=\frac{p_{i j}}{p_{j}}, i=1,2, \cdots, P\left(Y=y_{j} | X=x_{i}\right)=\frac{p_{ij}}{p_{i}}, j=1,2, \cdots$$

2, 连续性二维随机变量及其分布

分布函数    $$F(x, y)=\int_{-\infty}^{x} \int_{-\infty}^{y} f(u, v) d u d v$$

性质： $$F(+\infty,+\infty)=1, \frac{\partial^{2} F(x, y)}{\partial x \partial y}=f(x, y), P((x, y) \in G)=\iint f(x, y) d x d y$$

边缘分布函数与边缘密度函数

分布函数   $$F_{X}(x)=\int_{-\infty}^{x} \int_{-\infty}^{+\infty} f(u, v) d v d u$$                  密度函数         $$f_{X}(x)=\int_{-\infty}^{+\infty} f(x, v) d v$$  
$$F_{Y}(y)=\int_{-\infty}^{y} \int_{-\infty}^{+\infty} f(u, v) d u d v$$                                            $$f_{Y}(y)=\int_{-\infty}^{+\infty} f(u, y) d u$$

条件概率密度

$$f_{Y | X}(y | x)=\frac{f(x, y)}{f_{X}(x)},-\infty<y<+\infty$$          $$f_{X Y}(x | y)=\frac{f(x, y)}{f_{Y}(y)},-\infty<x<+\infty$$

二维随机变量和函数的分布

离散型  $$P\left(Z=z_{k}\right)=\sum_{x_i+y_j=z_k} P\left(X=x_{i}, Y=y_{j}\right)$$ 注意部分可加性

连续性  $$f_{z}(z)=\int_{-\infty}^{+\infty} f(x, z-x) d x=\int_{-\infty}^{\infty} f(z-y, y) d y$$

**常见二维随机变量的联合分布**

\(1\) 二维均匀分布：$$(x, y) \sim U(D), f(x, y)=\left\{\begin{array}{l}{\frac{1}{S(D)},(x, y) \in D} \\ 0, else \end{array}\right.$$

\(2\) 二维正态分布: $$(X, Y) \sim N\left(\mu_{1}, \mu_{2}, \sigma_{1}^{2}, \sigma_{2}^{2}, \rho\right),(X, Y) \sim N\left(\mu_{1}, \mu_{2}, \sigma_{1}^{2}, \sigma_{2}^{2}, \rho\right)$$

$$f(x, y)=\frac{1}{2 \pi \sigma_{1} \sigma_{2} \sqrt{1-\rho^{2}}} \cdot \exp \left\{\frac{-1}{2\left(1-\rho^{2}\right)}\left[\frac{\left(x-\mu_{1}\right)^{2}}{\sigma_{1}^{2}}-2 \rho \frac{\left(x-\mu_{1}\right)\left(y-\mu_{2}\right)}{\sigma_{1} \sigma_{2}}+\frac{\left(y-\mu_{3}\right)^{2}}{\sigma_{2}^{2}}\right]\right\}$$

## 随机变量的数字特征

1，数学期望

定义：离散型 $$E(X)=\sum_{k=1}^{+\infty} x_{k} p_{k}$$            连续型  $$E(X)=\int_{-\infty}^{+\infty} x f(x) d x$$

性质 $$E(C)=C, E[E(X)]=E(X), E(C X)=C E(X), E(X \pm Y)=E(X) \pm E(Y)$$ $$E(a X \pm b)=a E(X) \pm b$$

当X,Y相互独立时，$$E(X Y)=E(X) E(Y)$$

2，方差

定义：$$D(X)=E\left[(X-E(X))^{2}\right]=E\left(X^{2}\right)-E^{2}(X)$$

性质 $$D(C)=0, D(a X \pm b)=a^{2} D(X), \quad D(X \pm Y)=D(X)+D(Y) \pm 2Cov(X, Y)$$

当X, Y相互独立时 $$D(X \pm Y)=D(X)+D(Y)$$

3，协方差与相关系数

协方差 $$Cov\,(X, Y)=E(X Y)-E(X) E(Y)$$  当X,Y相互独立时 $$Cov\,(X, Y)=0$$

相关系数 $$\rho_{x \gamma}=\frac{Cov(X, Y)}{\sqrt{D(X)} \sqrt{D(Y)}}$$   当X,Y相互独立时 $$\rho_{X Y}=0$$ X,Y不相关

协方差和相关系数的性质 $$cov(X, X)=D(X), \quad Cov(X, Y)=Cov(Y, X)$$

$$cov\left(X_{1}+X_{2}, Y\right)=Cov\left(X_{1}, Y\right)+Cov\left(X_{2}, Y\right), \quad Cov(a X+c, b Y+\alpha)=a b Cov(X, Y)$$

$$Cov(x, a)=0$$ a为常数   $$D(a X \pm b Y)=a^{2} D(X)+b^{2} D(Y) \pm 2 a b Cov(X, Y)$$

4，常见随机变量分布的数学期望和方差

![](/assets/normal-expandvaiance.png)

## 大数定律和中心极限定理

## 数理统计的基本概念

1，总体和样本的分布函数

设总体$$X\sim F(x)$$,则样本的联合分布函数$$F\left(x_{1}, x_{2} \cdots x_{n}\right)=\prod_{k=1}^{n} F\left(x_{\hat{\varepsilon}}\right)$$

2， 统计量

样本均值 $$\overline{X}=\frac{1}{n} \sum_{i=1}^{n} X_{i}$$   样本方差 $$S^{2}=\frac{1}{n-1} \sum_{i=1}^{n}\left(X_{i}-\overline{X}\right)^{2}=\frac{1}{n-1} \sum_{i=1}^{n}\left(X_{i}^{2}-n \overline{X}^{2}\right)$$

样本标准差 $$S=\sqrt{\frac{1}{n-1} \sum_{i=1}^{n}\left(X_{i}-\overline{X}\right)^{2}}$$   样本的k阶原点距 $$A_{\mathrm{k}}=\frac{1}{n} \sum_{i=1}^{n} X_{i}^{k}, k=1,2...$$

样本的k阶中心距 $$B_{k}=\frac{1}{n} \sum_{i=1}^{\kappa}\left(X_{i}-\overline{X}\right)^{*}, k=1,2,3 \cdots$$

3，三大抽样分布

a, $$\chi^2$$分布：设随机变量 $$\mathrm{X} \sim \mathrm{B}(0,1)(i=1,2, \cdots, n)$$且相互独立，则称统计量

$$\chi^2=X_1^2+X_2^2+...X_n^2$$服从自由度为n的$$\chi^2$$分布，记为$$\chi^2=\chi^2(n)$$

性质  $$E\left[\chi^{2}(n)\right]=n, D\left[\chi^{2}(n)\right]=2 n$$  设$$X \sim \chi^{2}(m), Y \sim \chi^{2}(n)$$ 且相互独立，则$$X+Y \sim \chi^{2}(m+n)$$

b,t分布：设随机变量$$X \sim N(0,1), Y \sim \chi^{2}(n)$$, 且X与Y独立，则称统计量

$$T=\frac{X}{\sqrt{Y / n}}$$服从自由度为n的t分布， 记为$$T \sim t(n)$$

性质 $$E(T)=0(n>1), D(T)=\frac{n}{n-2}(n>2)$$

$$\lim _{n \rightarrow \infty} f_{n}(x)=\varphi(x)=\frac{1}{\sqrt{2 \pi}} e^{-\frac{x^{2}}{2}}$$

c,F分布，设随机变量 $$X \sim \chi^{2}(n_1), Y \sim \chi^{2}(n_2)$$ 且X与Y相互独立，则称随机变量

$$F=\frac{\sum_{i=1}^{n_{1}} X_{i}^{2}}{n_{1}} / \frac{\sum_{i=1}^{n_{2}} Y_{i}^{2}}{n_{2}}$$ 服从自由度n1和n2的F分布。

## Information Theory {#information-theory}

信息理论是应用数学的一个分支，主要是来围绕着定量的分析a signal中包含了多少信息。最开始是用来研究通过噪声通道发送离散字母的信息发送问题，例如通过无线电的通信。在这个context中，信息论能告诉我们怎么就优化编码，计算预期的消息长度并从特定采样的概率分布使用不同的编码方案，在机器学习当中，可以将信息论应用于连续变量，消息的长度解释（message length interpretations）并未应用于这些变量的情况下\*\*  
采用这种正式的直觉（intuition）

* 相似的事件拥有较少的信息量，在极端情况下，事件确定发生的情况，应该不包含
* 相似度小的事情应该拥有更高的信息量
* 独立事件应该有附加信息

为了满足这三个性质，所以呢定义了self-information I\(X\)=-LogP\(x\)（自信息（英语：self-information），又译为信息本体，由克劳德·香农提出，用来衡量单一事件发生时所包含的信息量多寡）。用nat作为单位，1nat就是观测可能性为1/e的事件所包含的信息量。可以用Shannon entropy（香农熵）来衡量概率分布当中的不确定性大小。当变量是连续变量时，香龙熵就变成了（differencial entropy）微分熵。如果说遇到了两个分布，那么用Kullback-Leibler \(KL\) divergence来衡量两个分布之间的不同。

# Structured Probabilistic Models-结构化的概率模型 {#structured-probabilistic-models-结构化的概率模型}

机器学习通常涉及非常大量的随机变量的概率分布。通常，这些概率分布会涉及相对较少变量的直接相互作用。采用一个单一的函数来描述整个联合概率分布式非常有效的（不论是计算效率还是统计效率）。当我们用一个graph来表征概率分布的factorization，就可以称之为structured probabilistic model 结构化概率模型 or graphical model 图论模型。结构概率模型分为两种：直接和间接。这两种方法都是用一个图，每个图的节点表示一个变量，连接点之间的边表示这两个随机变量之间的直接相关的概率分布。

