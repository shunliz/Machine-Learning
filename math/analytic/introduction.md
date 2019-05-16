本章主要回顾数学分析的主要知识点。专题介绍机器学习中常用的一些方法。

# 导数定义

导数和微分的概念

$$f^{\prime}\left(x_{0}\right)=\lim _{\Delta x \rightarrow 0} \frac{f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right)}{\Delta x}$$                                 \(1\)

或者

$$f^{\prime}\left(x_{0}\right)=\lim _{x \rightarrow x_{0}} \frac{f(x)-f\left(x_{0}\right)}{x-x_{0}}$$                                           \(2\)

# **函数的可导性与连续性之间的关系**

**Th1: **函数f\(x\)在$$x_0$$处可微$$\Leftrightarrow f(x)$$在$$x_0$$处可导

**Th2: **若函数在点$$x_0$$处可导，则y=f\(x\)在点$$x_0$$处连续，反之则不成立。即函数连续不一定可导。

**Th3: **$$f^{\prime}\left(x_{0}\right)$$存在$$\Leftrightarrow f^{\prime}_{-}\left(x_{0}\right)=f^{\prime}_{+}\left(x_{0}\right)$$

# **平面曲线的切线和法线**

切线方程 :$$y-y_{0}=f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)$$

法线方程：$$y-y_{0}=-\frac{1}{f^{\prime}\left(x_{0}\right)}\left(x-x_{0}\right), f^{\prime}\left(x_{0}\right) \neq 0$$

# **四则运算法则**

设函数u=u\(x\), v=v\(x\)在点x可导:则

$$(1)(u \pm v)^{\prime}=u^{\prime} \pm v^{\prime}$$                                       $$d(u \pm v)=d u \pm d v$$

$$(2)(u v)^{\prime}=u v^{\prime}+v u^{\prime}$$                                        $$d(u v)=u d v+v d u$$

$$(3)\left(\frac{u}{v}\right)^{\prime}=\frac{v u^{\prime}-u v^{\prime}}{v^{2}}(v \neq 0)$$                                $$d\left(\frac{u}{v}\right)=\frac{v d u-u d v}{v^{2}}$$

# **基本导数与微分表**

1. $$y=c$$                   $$y^{\prime}=0 $$                    $$d y=0$$
2. $$y=x^{\alpha}$$                $$y^{\prime}=\alpha x^{\alpha-1}$$          $$d y=\alpha x^{\alpha-1} d x$$
3. $$y=a^{x} $$                 $$y^{\prime}=a^{x}   \ln a$$         $$ d y=a^{x} \ln a d x$$                       特例  $$\left(e^{x}\right)^{\prime}=e^{x} $$     $$d\left(e^{x}\right)=e^{x} d x$$
4. $$y=\ln _{a} x$$            $$y^{\prime}=\frac{1}{x \ln a}$$            $$d y=\frac{1}{x \ln a} d x$$                          特例   $$y=\ln x$$        $$(\ln x)^{\prime}=\frac{1}{x}$$              $$d(\ln x)=\frac{1}{x} d x$$
5. $$y=\sin x$$             $$y^{\prime}=\cos x$$             $$d(\sin x)=\cos x d x$$
6. $$y=\cos x$$             $$y^{\prime}=-\sin x$$         $$d(\cos x)=-\sin x d x$$
7. $$y=\tan x$$             $$y^{\prime}=\frac{1}{\cos ^{2} x}=\sec ^{2} x$$     $$d(\tan x)=\sec ^{2} x d x$$
8. $$y=\cot x$$              $$y^{\prime}=-\frac{1}{\sin ^{2} x}=-\csc ^{2} x$$          $$d(\cot x)=-\csc ^{2} x d x$$
9. $$y=\sec x$$               $$y^{\prime}=\sec x \tan x$$               $$d(\sec x)=\sec x \tan x d x$$
10. $$y=\csc x$$               $$y^{\prime}=-\csc x \cot x$$             $$d(\csc x)=-\csc x \cot x d x$$
11. $$y=\arcsin x$$          $$y^{\prime}=\frac{1}{\sqrt{1-x^{2}}}$$                        $$d(\arcsin x)=\frac{1}{\sqrt{1-x^{2}}} d x$$
12. $$y=\arccos x$$          $$y^{\prime}=-\frac{1}{\sqrt{1-x^{2}}}$$                     $$d(\arccos x)=-\frac{1}{\sqrt{1-x^{2}}} d x$$
13. $$y=\arctan x$$          $$y^{\prime}=\frac{1}{1+x^{2}}$$                            $$d(\arctan x)=\frac{1}{1+x^{2}} d x$$

# **复合函数，反函数，隐函数以及参数方程所确定的函数的微分法**

\(1\) 反函数的运算法则: 设y=f\(x\)在点x的某邻域内单调连续，在点x处可导且$$f^{\prime}(x) \neq 0$$，则其反函数在点x所对应的y处可导，并且有$$\frac{d y}{d x}=\frac{1}{\frac{d x}{d y}}$$

\(2\) 复合函数的运算法则:若$$\mu=\varphi(x)$$在点x可导,而$$y=f(\mu)$$在对应点$$\mu(\mu=\varphi(x))$$可导,则复合函数$$y=f(\varphi(x))$$在点x可导,且$$y^{\prime}=f^{\prime}(\mu) \cdot \varphi^{\prime}(x)$$

\(3\) 隐函数导数$$\frac{d y}{d x}$$的求法一般有三种方法：

1\)方程两边对x求导，要记住y是x的函数，则y的函数是x的复合函数.例如$$\frac{1}{y}, y^{2}, \ln y, e^{y}$$等均是x的复合函数. 对x求导应按复合函数连锁法则做.

2\)公式法.由F\(x,y\)=0 知$$\frac{d y}{d x}=-\frac{F_{x}^{\prime}(x, y)}{F_{y}^{\prime}(x, y)}$$,其中，$$F_{x}^{\prime}(x, y), \quad F_{y}^{\prime}(x, y)$$分别表示F\(x,y\)对x和y的偏导数

3\)利用微分形式不变性

# **常用高阶导数公式**

1. $$\left(a^{x}\right)^{(n)}=a^{x} \ln ^{n} a \quad(a>0)$$
2. $$\left(e^{x}\right)^{(n)}=e^{x}$$
3. $$(\sin k x)^{(n)}=k^{n} \sin \left(k x+n \cdot \frac{\pi}{2}\right)$$
4. $$(\cos k x)^{(n)}=k^{n} \cos \left(k x+n \cdot \frac{\pi}{2}\right)$$
5. $$(\ln x)^{(n)}=(-1)^{(n-1)} \frac{(n-1) !}{x^{n}}$$
6. 莱布尼兹公式：若u\(x\),v\(x\)均n阶可导，则$$(u v)^{(n)}=\sum_{i=0}^{n} c_{n}^{i} u^{(i)} v^{(n-i)}$$，其中$$u^{(0)}=u, v^{(0)}=v$$

# **方向导数**

$$\frac{\partial f}{\partial l}=\frac{\partial f}{\partial x} \cos \varphi+\frac{\partial f}{\partial y} \sin \varphi$$

# **微分中值定理，泰勒公式**

**Th1:\(费马定理\)**

若函数f\(x\)满足条件： \(1\)函数f\(x\)在$$x_0$$的某邻域内有定义，并且在此邻域内恒有$$f(x) \leq f\left(x_{0}\right)$$或$$f(x) \geq f\left(x_{0}\right)$$,

\(2\)f\(x\)在$$x_0$$处可导,则有$$f^{\prime}\left(x_{0}\right)=0$$

**Th2:\(罗尔定理\)**

设函数f\(x\)满足条件：

\(1\)在闭区间\[a,b\]上连续；

\(2\)在\(a,b\)内可导；

\(3\)f\(a\)=f\(b\)

则在\(a,b\)内存在一个$$\xi$$，使$$f^{\prime}(\xi)=0$$

**Th3:\(拉格朗日中值定理\)**

设f\(x\)函数满足条件：

\(1\)在\[a,b\]上连续；

\(2\)在\(a,b\)内可导；

则在\(a,b\)内存在一个$$\xi$$，使$$\frac{f(b)-f(a)}{b-a}=f^{\prime}(\xi)$$

**Th4:\(柯西中值定理\)**

设函数f\(x\)，g\(x\)满足条件：

\(1\) 在\[a,b\]上连续；

\(2\) 在\(a,b\)内可导且$$f^{\prime}(x), g^{\prime}(x)$$均存在，且$$g^{\prime}(x) \neq 0$$

则在\(a,b\)内存在一个$$\xi$$，使$$\frac{f(b)-f(a)}{g(b)-g(a)}=\frac{f^{\prime}(\xi)}{g^{\prime}(\xi)}$$

# **洛必达法则**

**法则Ⅰ \(**$$\frac{0}{0}$$**型\) **

设函数f\(x\), g\(x\)满足条件：$$\lim _{x \rightarrow x_{0}} f(x)=0, \lim _{x \rightarrow x_{0}} g(x)=0$$;

f\(x\), g\(x\)在$$x_0$$的邻域内可导，\(在$$x_0$$处可除外\)且$$g^{\prime}(x) \neq 0$$;

$$\lim _{x \rightarrow x_{0}} \frac{f^{\prime}(x)}{g^{\prime}(x)}$$存在\(或$$\infty$$\)。

则:$$\lim _{x \rightarrow x_{0}} \frac{f(x)}{g(x)}=\lim _{x \rightarrow x_{0}} \frac{f^{\prime}(x)}{g^{\prime}(x)}$$。

**法则**$$I^{\prime}$$**\(**$$\frac{0}{0}$$**型\)**

设函数f\(x\),g\(x\)满足条件：$$\lim _{x \rightarrow \infty} f(x)=0, \lim _{x \rightarrow \infty} g(x)=0$$;

存在一个X&gt;0,当$$|x|>X$$时,f\(x\),g\(x\)可导,且$$g^{\prime}(x) \neq 0 ; \lim _{x \rightarrow x_{0}} \frac{f^{\prime}(x)}{g^{\prime}(x)}$$;存在\(或$$\infty$$\)。

则:$$\lim _{x \rightarrow x_{0}} \frac{f(x)}{g(x)}=\lim _{x \rightarrow x_{0}} \frac{f^{\prime}(x)}{g^{\prime}(x)}$$

**法则Ⅱ\(**$$\frac{\infty}{\infty}$$**型\) **

设函数f\(x\), g\(x\)满足条件：

$$\lim _{x \rightarrow x_{0}} f(x)=\infty, \lim _{x \rightarrow x_{0}} g(x)=\infty ; f(x), g(x)$$在$$x_0$$的邻域内可导\(在$$x_0$$处可除外\)且$$g^{\prime}(x) \neq 0$$;

$$\lim _{x \rightarrow x_{0}} \frac{f^{\prime}(x)}{g^{\prime}(x)}$$存在\(或$$\infty$$\)。则$$\lim _{x \rightarrow x_{0}} \frac{f(x)}{g(x)}=\lim _{x \rightarrow x_{0}} \frac{f^{\prime}(x)}{g^{\prime}(x)}$$

同理法则$$I I^{\prime}$$\($$\frac{\infty}{\infty}$$型\)仿法则$$I^{\prime}$$可写出。

# **泰勒公式**

设函数f\(x\)在点$$x_0$$处的某邻域内具有n+1阶导数，则对该邻域内异于$$x_0$$的任意点x，在$$x_0$$与x之间至少存在 一个$$\xi$$，使得$$f(x)=f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)+\frac{1}{2 !} f^{\prime \prime}\left(x_{0}\right)\left(x-x_{0}\right)^{2}+\cdots+\frac{f^{(n)}\left(x_{0}\right)}{n !}\left(x-x_{0}\right)^{n}+R_{n}(x)$$：其中$$R_{n}(x)=\frac{f^{(n+1)}(\xi)}{(n+1) !}\left(x-x_{0}\right)^{n+1}$$称为f\(x\)在点$$x_0$$处的n阶泰勒余项。

令$$x_0=0$$，则n阶泰勒公式$$f(x)=f(0)+f^{\prime}(0) x+\frac{1}{2 !} f^{\prime \prime}(0) x^{2}+\cdots+\frac{f^{(n)}(0)}{n !} x^{n}+R_{n}(x)$$ .........\(1\)其中$$R_{n}(x)=\frac{f^{(n+1)}(\xi)}{(n+1) !} x^{n+1}$$，$$\xi$$在0与x之间.式称为麦克劳林公式

**常用五种函数在**$$x_0=0$$**处的泰勒公式**

1. $$e^{x}=1+x+\frac{1}{2 !} x^{2}+\cdots+\frac{1}{n !} x^{n}+\frac{x^{n+1}}{(n+1) !} e^{\xi}$$  或者 $$e^x=1+x+\frac{1}{2 !} x^{2}+\cdots+\frac{1}{n !} x^{n}+o\left(x^{n}\right)$$
2. $$\sin x=x-\frac{1}{3 !} x^{3}+\cdots+\frac{x^{n}}{n !} \sin \frac{n \pi}{2}+\frac{x^{n+1}}{(n+1) !} \sin \left(\xi+\frac{n+1}{2} \pi\right)$$ 或者 $$sinx=x-\frac{1}{3 !} x^{3}+\cdots+\frac{x^{n}}{n !} \sin \frac{n \pi}{2}+o\left(x^{n}\right)$$
3. $$\cos x=1-\frac{1}{2 !} x^{2}+\cdots+\frac{x^{n}}{n !} \cos \frac{n \pi}{2}+\frac{x^{n+1}}{(n+1) !} \cos \left(\xi+\frac{n+1}{2} \pi\right)$$
4. $$\ln (1+x)=x-\frac{1}{2} x^{2}+\frac{1}{3} x^{3}-\cdots+(-1)^{n-1} \frac{x^{n}}{n}+\frac{(-1)^{n} x^{n+1}}{(n+1)(1+\xi)^{n+1}}$$
5. $$(1+x)^{m}=1+m x+\frac{m(m-1)}{2 !} x^{2}+\cdots+\frac{m(m-1) \cdots(m-n+1)}{n !} x^{n}+\frac{m(m-1) \cdots(m-n+1)}{(n+1) !} x^{n+1}(1+\xi)^{m-n-1}$$

# **函数单调性的判断**

**Th1:**设函数f\(x\)在区间\(a,b\)内可导，如果对$$\forall x \in(a, b)$$，都有$$f^{\prime}(x)>0$$（或$$f^{\prime}(x)<0$$），则函数f\(x\)在\(a,b\)内是单调增加的（或单调减少）

**Th2:**（取极值的必要条件）设函数f\(x\)在$$x_0$$处可导，且在$$x_0$$处取极值，则$$f^{\prime}\left(x_{0}\right)=0$$。

**Th3:**（取极值的第一充分条件）设函数f\(x\)在$$x_0$$的某一邻域内可微，且$$f^{\prime}\left(x_{0}\right)=0$$（或f\(x\)在$$x_0$$处连续，但$$f^{\prime}\left(x_{0}\right)$$不存在。）

\(1\)若当x经过$$x_0$$时，f\(x\)由“+”变“-”，则为极大值； \(2\)若当x经过$$x_0$$时，由“-”变“+”，则为极小值； \(3\)若$$f^{\prime}(x)$$经过$$x = x_0$$的两侧不变号，则不是极值。

**Th4:**\(取极值的第二充分条件\)设f\(x\)在点$$x_0$$处有$$f^{\prime \prime}(x) \neq 0$$，且$$f^{\prime}\left(x_{0}\right)=0$$，则$$f^{\prime \prime} \left(x_{0}\right)<0$$ 当时，$$f(x_0)$$为极大值； 当$$f^{\prime \prime}\left(x_{0}\right)>0$$时，$$f(x_0)$$为极小值。 注：如果$$f^{\prime \prime}\left(x_{0}\right)=0$$，此方法失效。

# **函数凹凸性的判断**

**Th1:**\(凹凸性的判别定理）若在I上$$f^{\prime \prime}(x)<0$$（或$$f^{\prime \prime}(x)>0$$），则f\(x\)在I上是凸的（或凹的）。

**Th2:**\(拐点的判别定理1\)若在$$x_0$$处$$f^{\prime \prime}(x)=0$$，（或$$f^{\prime \prime}(x)$$不存在），当x变动经过$$x_0$$时，$$f^{\prime \prime}(x)$$变号，则$$(x_0, f(x_0))$$为拐点。

**Th3:**\(拐点的判别定理2\)设f\(x\)在点$$x_0$$的某邻域内有三阶导数，且$$f^{\prime \prime}(x)=0$$，$$f^{\prime \prime \prime}(x) \neq 0$$，则$$(x_0,f(x_0))$$为拐点。

# Jensen不等式：

若f是凸函数

$$\begin{array}{l}{\theta_{1}, \ldots, \theta_{k} \geq 0, \theta_{1}+\cdots+\theta_{k}=1} \\ {f\left(\theta_{1} x_{1}+\cdots+\theta_{k} x_{k}\right) \leq \theta_{1} f\left(x_{1}\right)+\cdots+\theta_{k} f\left(x_{k}\right)}\end{array}$$

$$p(x) \geq 0 \text { on } S \subseteq dom f, \int_{S} p(x) d x=1$$

$$f\left(\int_{S} p(x) x d x\right) \leq \int_{S} f(x) p(x) d x$$

$$f(\mathbf{E} x) \leq \mathbf{E} f(x)$$

# **弧微分**

$$d S=\sqrt{1+y^{\prime 2}} d x$$

# **曲率**

曲线y=f\(x\)在点\(x,y\)处的曲率$$k=\frac{\left|y^{\prime \prime}\right|}{\left(1+y^{\prime}\right)^{\frac{3}{2}}}$$。 对于参数方程$$\left\{\begin{array}{l}{x=\varphi(t)} \\ {y=\psi(t)}\end{array}\right., k=\frac{\left|\varphi^{\prime}(t) \psi^{\prime \prime}(t)-\varphi^{\prime \prime}(t) \psi^{\prime}(t)\right|}{\left[\varphi^{\prime 2}(t)+\psi^{2}(t)\right]^{\frac{3}{2}}}$$

**曲率半径**

曲线在点M处的曲率$$k(k \neq 0)$$与曲线在点M处的曲率半径$$\rho$$有如下关系：$$\rho=\frac{1}{k}$$

# 微分应用

1， 已知函数f\(x\)=x^x，x&gt;0,求f\(x\)的最小值

$$\begin{array}{l}{t(x)=x^{x}} \\ {\Rightarrow \ln t=x \ln x}\end{array}$$

两边对x求导 $$\frac{1}{t} t^{\prime}=\ln x+1$$

令$$t'=0$$,        $$\ln x+1=0$$

$$\begin{array}{l}{\Rightarrow x=e^{-1}} \\ {\Rightarrow t=e^{-\frac{1}{e}}}\end{array}$$

# 积分应用

$$N \rightarrow \infty \Rightarrow \ln N ! \rightarrow N(\ln N-1)$$

$$\begin{array}{l}{\ln N !=\sum_{i=1}^{N} \ln i \approx \int_{1}^{N} \ln x d x} \\ {=x \ln \left.x\right|_{1} ^{N}-\int_{1}^{N} x d \ln x} \\ {=N \ln N-\int_{1}^{N} x \cdot \frac{1}{x} d x} \\ {=N \ln N-\left.x\right|_{1} ^{N}} \\ {=N \ln N-N+1} \\ {\rightarrow N \ln N-N}\end{array}$$

# **最优化**

**机器学习 = 模型 + 策略 + 算法**

可以看得出，算法在机器学习中的 重要性。实际上，这里的算法指的就是**优化算法.**

## 最优化问题的数学描述

最优化的基本数学模型如下：
$$\min f(\mathbf{x})$$
$$\begin{aligned} \text { s.t. } & h_{i}(\mathbf{x})=0 \\ & g_{j}(\mathbf{x}) \leqslant 0 \end{aligned}$$

它有三个基本要素，即：

* 设计变量：x是一个实数域范围内的n维向量，被称为决策变量或问题的解；
* 目标函数：f\(x\)为目标函数；
* 约束条件：$$h_i(x)=0$$称为等式约束，$$g_j(x) \leq 0$$为不等式约束，$$i=0,1,2,\dots$$

## **凸集与凸集分离定理**

### **1、凸集**

实数域R上（或复数C上）的向量空间中，如果集合S中任两点的连线上的点都在S内，则称集合S为凸集，如下图所示：![](/assets/convetex1.png)

**数学定义为：**

设集合$$D \subset R^{n}$$，若对于任意两点$$x, y \in D$$，及实数$$\lambda(0 \leq \lambda \leq 1)$$都有：$$\lambda x+(1-\lambda) \,\,\,\,\ y \in D$$则称集合D为凸集。

**2、超平面和半空间**

实际上，二维空间的超平面就是一条线（可以使曲线），三维空间的超平面就是一个面（可以是曲面）。其数学表达式如下：

**超平面：**$$H=\left\{x \in R^{n} | a_{1}+a_{2}+\ldots+a_{n}=b\right\}$$

**半空间：**$$H^{+}=\left\{x \in R^{n} | a_{1}+a_{2}+\ldots+a_{n} \geq b\right\}$$

**3、凸集分离定理**

所谓两个凸集分离，直观地看是指两个凸集合没有交叉和重合的部分，因此可以用一张超平面将两者隔在两边，如下图所示：

![](/assets/convetex2.png)

**4、凸函数**

凸函数就是一个定义域在某个向量空间的凸子集C上的实值函数。

![](/assets/convetex3.png)

**数学定义为：**

对于函数f\(x\)，如果其定义域C是凸的，且对于∀x,y∈C，$$0 \leq \alpha \leq 1$$，有：

$$f(\theta x+(1-\theta) y) \leq \theta f(x)+(1-\theta) f(y)$$

则f\(x\)是凸函数。

**注：**如果一个函数是凸函数，则其局部最优点就是它的全局最优点。这个性质在机器学习算法优化中有很重要的应用，因为机器学习模型最后就是在求某个函数的全局最优点，一旦证明该函数（机器学习里面叫“损失函数”）是凸函数，那相当于我们只用求它的局部最优点了。  
  




