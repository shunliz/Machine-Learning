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
14. $$y=\operatorname{arccot} x$$           $$y^{\prime}=-\frac{1}{1+x^{2}}$$                         $$d(\operatorname{arccot} x)=-\frac{1}{1+x^{2}} d x$$

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



