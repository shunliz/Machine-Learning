# 线性代数

## 行列式

### **1.行列式按行（列）展开定理**

\(1\) 设$$A=\left(a_{i j}\right)_{n \times n}$$，则：$$a_{i 1} A_{j 1}+a_{i 2} A_{j 2}+\cdots+a_{i n} A_{j n}=\left\{\begin{array}{l}{|A|, i=j} \\ {0, i \neq j}\end{array}\right.$$ 或$$a_{1 i} A_{1 j}+a_{2 i} A_{2 j}+\cdots+a_{n i} A_{n j}=\left\{\begin{array}{l}{|A|, i=j} \\ {0, i \neq j}\end{array}\right.$$

即$$A A^{*}=A^{*} A=|A| E$$ 其中;

$$A^{*}=\left( \begin{array}{cccc}{A_{11}} & {A_{12}} & {\ldots} & {A_{1 n}} \\ {A_{21}} & {A_{22}} & {\dots} & {A_{2 n}} \\ {\ldots} & {\ldots} & {\cdots} & {\cdots} \\ {A_{n 1}} & {A_{n 2}} & {\ldots} & {A_{n n}}\end{array}\right)=\left(A_{j i}\right)=\left(A_{i j}\right)^{T}$$

$$D_{n}=\left| \begin{array}{cccc}{1} & {1} & {\ldots} & {1} \\ {x_{1}} & {x_{2}} & {\ldots} & {x_{n}} \\ {\ldots} & {\ldots} & {\ldots} & {\ldots} \\ {x_{1}^{n-1}} & {x_{2}^{n-1}} & {\ldots} & {x_{n}^{n-1}}\end{array}\right|=\prod_{1 \leq j<i \leq n}\left(x_{i}-x_{j}\right)$$

\(2\) 设A,B为n阶方阵，则$$| | A B|=| A| | B|=| B| | A|=| B A |$$，但$$|A \pm B|=|A| \pm|B|$$不一定成立。

\(3\)$$|k A|=k^{n}|A|_{,} A$$为n阶方阵。

\(4\) 设A为n阶方阵，$$\left|A^{T}\right|=|A| ;\left|A^{-1}\right|=|A|^{-1}$$（若A可逆），$$\left|A^{*}\right|=|A|^{n-1}$$ $$n \geq 2$$

\(5\)$$\left| \begin{array}{cc}{A} & {O} \\ {O} & {B}\end{array}\right|=\left| \begin{array}{cc}{A} & {C} \\ {O} & {B}\end{array}\right|=\left| \begin{array}{cc}{A} & {O} \\ {C} & {B}\end{array}\right|=|A||B|, A, B$$为方阵，但$$\left| \begin{array}{cc}{O} & {A_{m \times m}} \\ {B_{n \times n}} & {O}\end{array}\right|=(-1)^{m n}|A||B|$$。

\(6\) 范德蒙行列式$$D_{n}=\left| \begin{array}{cccc}{1} & {1} & {\dots} & {1} \\ {x_{1}} & {x_{2}} & {\dots} & {x_{n}} \\ {\ldots} & {\ldots} & {\ldots} & {\ldots} \\ {x_{1}^{n-1}} & {x_{2}^{n 1}} & {\ldots} & {x_{n}^{n-1}}\end{array}\right|=\prod_{1 \leq j<i \leq n}\left(x_{i}-x_{j}\right)$$

设A是n阶方阵，$$\lambda_{i}(i=1,2 \cdots, n)$$是A的n个特征值，则$$|A|=\prod_{i=1}^{n} \lambda_{i}$$

## 矩阵

矩阵：mxn个数$$a_{ij}$$排成m行n列的表格$$\left[ \begin{array}{cccc}{a_{11}} & {a_{12}} & {\cdots} & {a_{1 n}} \\ {a_{21}} & {a_{22}} & {\cdots} & {a_{2 n}} \\ {\cdots} & {\cdots} & {\cdots} & {\cdots} \\ {a_{m 1}} & {a_{m 2}} & {\cdots} & {a_{m n}}\end{array}\right]$$称为矩阵，简记为A，或者$$\left(a_{i j}\right)_{m \times n}$$。若m=n，则称A是n阶矩阵或n阶方阵。

### **矩阵的线性运算**

#### **1.矩阵的加法**

设$$A=\left(a_{i j}\right), B=\left(b_{i j}\right)$$是两个mxn矩阵，则mxn矩阵$$C=c_{i j} =a_{i j}+b_{i j}$$称为矩阵A与B的和，记为A+B=C。

**2.矩阵的数乘**

设$$A=\left(a_{i j}\right)$$是mxn矩阵，k是一个常数，则mxn矩阵$$ka_{ij}$$称为数k与矩阵A的数乘，记为kA。

**3.矩阵的乘法**

设$$A=\left(a_{i j}\right)$$是mxn矩阵，$$B=\left(b_{i j}\right)$$是nxs矩阵，那么mxs矩阵$$C=\left(c_{i j}\right)$$，其中$$c_{i j}=a_{i 1} b_{1 j}+a_{i 2} b_{2 j}+\cdots+a_{i n} b_{n j}=\sum_{k=1}^{n} a_{i k} b_{k j}$$称为AB的乘积，记为C=AB。

**4.**$$A^T$$**、**$$A^{-1}$$**、**$$A^*$$**三者之间的关系**

1. $$\left(A^{T}\right)^{T}=A,(A B)^{T}=B^{T} A^{T},(k A)^{T}=k A^{T},(A \pm B)^{T}=A^{T} \pm B^{T}$$
2. $$\left(A^{-1}\right)^{-1}=A,(A B)^{-1}=B^{-1} A^{-1},(k A)^{-1}=\frac{1}{k} A^{-1}$$  但 $$(A \pm B)^{-1}=A^{-1} \pm B^{-1}$$不一定成立。
3. $$\left(A^{*}\right)^{*}=|A|^{n-2} A(n \geq 3),(A B)^{*}=B^{*} A^{*},(k A)^{*}=k^{n-1} A^{*}(n \geq 2)$$ 但 $$(A \pm B)^{*}=A^{*} \pm B^{*}$$不一定成立。

4. $$\left(A^{-1}\right)^{T}=\left(A^{T}\right)^{-1},\left(A^{-1}\right)^{*}=\left(A A^{*}\right)^{-1},\left(A^{*}\right)^{T}=\left(A^{T}\right)^{*}$$

**5.有关**$$A^*$$**的结论**

1. $$A A^{*}=A^{*} A=|A| E$$
2. $$\left|A^{*}\right|=|A|^{n-1}(n \geq 2), \quad(k A)^{*}=k^{n-1} A^{*}, \quad\left(A^{*}\right)^{*}=|A|^{n-2} A(n \geq 3)$$
3. 若A可逆，则$$A^{*}=|A| A^{-1},\left(A^{*}\right)^{*}=\frac{1}{|A|} A$$

4. 若A为n阶方阵，则：$$r\left(A^{*}\right)=\left\{\begin{array}{ll}{n,} & {r(A)=n} \\ {1,} & {r(A)=n-1} \\ {0,} & {r(A)<n-1}\end{array}\right.$$

**6.有关**$$A^{-1}$$**的结论**

A可逆$$\Leftrightarrow A B=E ; \Leftrightarrow|A| \neq 0 ; \Leftrightarrow r(A)=n$$$$\Leftrightarrow A$$可以表示为初等矩阵的乘积；$$\Leftrightarrow A ; \Leftrightarrow A x=0$$。

**7.有关矩阵秩的结论**

1. 秩$$r(A)$$=行秩=列秩；

2. $$r\left(A_{m \times n}\right) \leq \min (m, n)$$

3. $$r\left(A_{m \times n}\right) \leq \min (m, n)$$

4. $$r(A \pm B) \leq r(A)+r(B)$$

5. 初等变换不改变矩阵的秩

6. $$r(A)+r(B)-n \leq r(A B) \leq \min (r(A), r(B))$$特别若$$A B=O$$则：$$r(A)+r(B) \leq n$$

7. 若$$A^{-1}$$存在$$\Rightarrow r(A B)=r(B)$$;若$$B^{-1}$$存在$$\Rightarrow r(A B)=r(A)$$. 若$$r\left(A_{m \times n}\right)=n \Rightarrow r(A B)=r(B)$$ 若$$r\left(A_{m \times s}\right)=n \Rightarrow r(A B)=r(A)$$。

8. $$r\left(A_{m \times s}\right)=n \Leftrightarrow A x=0$$只有零解

**8.分块求逆公式**

$$\begin{array}{ll}{\left( \begin{array}{cc}{A} & {O} \\ {O} & {B}\end{array}\right)^{-1}=\left( \begin{array}{cc}{A^{-1}} & {O} \\ {O} & {B^{-1}}\end{array}\right) ; \left( \begin{array}{cc}{A} & {C} \\ {O} & {B}\end{array}\right)^{-1}=\left( \begin{array}{cc}{A} & {B^{-1} C B^{-1}} \\ {O} & {B^{-1}}\end{array}\right)} \\ {\left( \begin{array}{cc}{A} & {O} \\ {C} & {B}\end{array}\right)^{-1}=\left( \begin{array}{cc}{A^{-1}} & {O} \\ {-B^{-1} C A^{-1}} & {B^{-1}}\end{array}\right) ; \left( \begin{array}{cc}{O} & {A} \\ {B} & {O}\end{array}\right)^{-1}=\left( \begin{array}{cc}{O} & {B^{-1}} \\ {A^{-1}} & {O}\end{array}\right)}\end{array}$$

这里A，B均为可逆方阵。

# 向量

## **1.有关向量组的线性表示**

1. $$\alpha_{1}, \alpha_{2}, \cdots, \alpha_{s}$$线性相关$$\Leftrightarrow$$至少有一个向量可以用其余向量线性表示。

2. $$\alpha_{2}, \cdots, \alpha_{s}$$线性无关，$$\alpha_{2}, \cdots, \alpha_{s}$$，$$\beta$$线性相关$$\Leftrightarrow$$$$\beta$$可以由$$\alpha_{2}, \cdots, \alpha_{s}$$唯一线性表示。

3. $$\beta$$可以由$$\alpha_{2}, \cdots, \alpha_{s}$$线性表示$$\Leftrightarrow r\left(\alpha_{1}, \alpha_{2}, \cdots, \alpha_{s}\right)=r\left(\alpha_{1}, \alpha_{2}, \cdots, \alpha_{s}, \beta\right)$$

## **2.有关向量组的线性相关性**

1. 部分相关，整体相关；整体无关，部分无关.

2. \(1\) n个n维向量$$\alpha_{1}, \alpha_{2} \cdots \alpha_{n}$$线性无关$$\Leftrightarrow\left|\left[\alpha_{1} \alpha_{2} \cdots \alpha_{n}\right]\right| \neq 0$$，n个n维向量$$\alpha_{1}, \alpha_{2} \cdots \alpha_{n}$$线性相关$$\Leftrightarrow\left|\left[\alpha_{1}, \alpha_{2}, \cdots, \alpha_{n}\right]\right|=0$$。 \(2\) n+1个n维向量线性相关。 \(3\)若$$\alpha_{2}, \cdots, \alpha_{s}$$线性无关，则添加分量后仍线性无关；或一组向量线性相关，去掉某些分量后仍线性相关。

## 3**.向量组的秩与矩阵的秩之间的关系**

设$$r\left(A_{m \times n}\right)=r$$，则A的秩r\(A\)与A的行列向量组的线性相关性关系为：

* 若$$r\left(A_{m \times n}\right)=r=m$$，则A的行向量组线性无关

* 若$$r\left(A_{m \times n}\right)=r<m$$，则A的行向量组线性相关

* 若$$r\left(A_{m \times n}\right)=r=n$$，则A的行向量组线性无关

* 若$$r\left(A_{m \times n}\right)=r<n$$，则A的行向量组线性相关

## 4**.n维向量空间的基变换公式及过渡矩阵**

若$$\alpha_{1}, \alpha_{2}, \cdots, \alpha_{n}$$与$$\beta_{1}, \beta_{2}, \cdots, \beta_{n}$$是向量空间V的两组基，则基变换公式为：

$$\left(\beta_{1}, \beta_{2}, \cdots, \beta_{n}\right)=\left(\alpha_{1}, \alpha_{2}, \cdots, \alpha_{n}\right) \left[ \begin{array}{cccc}{c_{11}} & {c_{12}} & {\cdots} & {c_{1 n}} \\ {c_{21}} & {c_{22}} & {\cdots} & {c_{2 n}} \\ {\ldots} & {\cdots} & {\cdots} & {\cdots} \\ {c_{n 1}} & {c_{n 2}} & {\cdots} & {c_{n n}}\end{array}\right]=\left(\alpha_{1}, \alpha_{2}, \cdots, \alpha_{n}\right) C$$

其中C是可逆矩阵，称为由基$$\alpha_{2}, \cdots, \alpha_{n}$$到基$$\beta_{2}, \cdots, \beta_{n}$$的过渡矩阵。

## 5**.坐标变换公式**

若向量$$\gamma$$在基$$\alpha_{2}, \cdots, \alpha_{n}$$与基$$\beta_{2}, \cdots, \beta_{n}$$的坐标分别是，$$X=\left(x_{1}, x_{2}, \cdots, x_{n}\right)^{T}$$,$$Y=\left(y_{1}, y_{2}, \cdots, y_{n}\right)^{T}$$

即：$$\gamma=x_{1} \alpha_{1}+x_{2} \alpha_{2}+\cdots+x_{n} \alpha_{n}=y_{1} \beta_{1}+y_{2} \beta_{2}+\cdots+y_{n} \beta_{n}$$，则向量坐标变换公式为X=CY或$$Y=C^{-1}X$$，其中C是从基$$\alpha_{1}, \alpha_{2}, \cdots, \alpha_{n}$$到基$$\beta_{2}, \cdots, \beta_{n}$$的过渡矩阵。

## 6**.向量的内积**

$$(\alpha, \beta)=a_{1} b_{1}+a_{2} b_{2}+\cdots+a_{n} b_{n}=\alpha^{T} \beta=\beta^{T} \alpha$$

## 7**.Schmidt正交化**

若$$\alpha_{1}, \alpha_{2}, \cdots, \alpha_{s}$$线性无关，则可构造$$\beta_{1}, \beta_{2}, \cdots, \beta_{s}$$使其两两正交，且$$\beta_i$$仅是$$\alpha_{1}, \alpha_{2}, \cdots, \alpha_{i}$$的线性组合$$(i=1,2, \cdots, n)$$，再把$$\beta_i$$单位化，记$$\gamma_{i}=\frac{\beta_{i}}{\left|\beta_{i}\right|}$$，则$$\gamma_{1}, \gamma_{2}, \cdots, \gamma_{i}$$是规范正交向量组。其中

$$\beta_{1}=\alpha_{1},\beta_{2}=\alpha_{2}-\frac{\left(\alpha_{2}, \beta_{1}\right)}{\left(\beta_{1}, \beta_{1}\right)} \beta_{1}, \quad \beta_{3}=\alpha_{3}-\frac{\left(\alpha_{3}, \beta_{1}\right)}{\left(\beta_{1}, \beta_{1}\right)} \beta_{1}-\frac{\left(\alpha_{3}, \beta_{2}\right)}{\left(\beta_{2}, \beta_{2}\right)} \beta_{2}$$

....................................................

$$\beta_{s}=\alpha_{s}-\frac{\left(\alpha_{s}, \beta_{1}\right)}{\left(\beta_{1}, \beta_{1}\right)} \beta_{1}-\frac{\left(\alpha_{s}, \beta_{2}\right)}{\left(\beta_{2}, \beta_{2}\right)} \beta_{2}-\dots-\frac{\left(\alpha_{s}, \beta_{s-1}\right)}{\left(\beta_{s-1}, \beta_{s-1}\right)} \beta_{s-1}$$

## 8**.正交基及规范正交基**

向量空间一组基中的向量如果两两正交，就称为正交基；若正交基中每个向量都是单位向量，就称其为规范正交基。

# 线性方程组

## **1．克莱姆法则**

线性方程组$$\left\{\begin{array}{c}{a_{11} x_{1}+a_{12} x_{2}+\cdots+a_{1 n} x_{n}=b_{1}} \\ {a_{21} x_{1}+a_{22} x_{2}+\cdots+a_{2 n} x_{n}=b_{2}} \\ {\ldots \ldots \ldots \ldots \ldots+a_{2 n} x_{n}=b_{2}} \\ {a_{n 1} x_{1}+a_{n 2} x_{2}+\cdots+a_{n n} x_{n}=b_{n}}\end{array}\right.$$，如果系数行列式$$D=|A| \neq 0$$，则方程组有唯一解，$$x_{1}=\frac{D_{1}}{D}, x_{2}=\frac{D_{2}}{D}, \cdots, x_{n}=\frac{D_{n}}{D}$$，其中$$D_j$$是把D中第j列元素换成方程组右端的常数列所得的行列式。

## **2.**

**n**阶矩阵A可逆$$\Leftrightarrow A x=0$$只有零解$$\Leftrightarrow \forall b, A x=b$$总有唯一解，一般地，$$r\left(A_{m \times n}\right)=n \Leftrightarrow A x=0$$只有零解。

## **3.非奇次线性方程组有解的充分必要条件，线性方程组解的性质和解的结构**

\(1\) 设A为mxn矩阵，若$$r\left(A_{m \times n}\right)=m$$，则对$$Ax = b$$而言必有$$r(A)=r(A : b)=m$$，从而$$A x=b$$有解。

\(2\) 设$$x_{1}, x_{2}, \cdots \cdot x_{s}$$为$$A x=b$$的解，则$$k_{1} x_{1}+k_{2} x_{2} \cdots+k_{s} x_{s}$$当$$k_{1}+k_{2}+\cdots+k_{s}=1$$时仍为$$Ax = b$$的解；但当$$k_{1}+k_{2}+\cdots+k_{s}=0$$时，则为$$Ax = 0$$的解。特别$$\frac{x_{1}+x_{2}}{2}$$为$$Ax = b$$的解；$$2 x_{3}-\left(x_{1}+x_{2}\right)$$为$$Ax = 0$$的解。

\(3\) 非齐次线性方程组$$Ax=b$$无解$$\Leftrightarrow r(A)+1=r(A) \Leftrightarrow b$$不能由的A列向量$$\alpha_{1}, \alpha_{2}, \cdots, \alpha_{n}$$线性表示。

## **4.奇次线性方程组的基础解系和通解，解空间，非奇次线性方程组的通解**

\(1\) 齐次方程组$$Ax = 0$$恒有解\(必有零解\)。当有非零解时，由于解向量的任意线性组合仍是该齐次方程组的解向量，因此$$Ax = 0$$的全体解向量构成一个向量空间，称为该方程组的解空间，解空间的维数是$$n-r(A)$$，解空间的一组基称为齐次方程组的基础解系。

\(2\)$$\eta_{1}, \eta_{2}, \cdots, \eta_{t}$$是$$Ax=0$$的基础解系，即：

1\)$$\eta_{1}, \eta_{2}, \cdots, \eta_{t}$$是$$Ax = 0$$的解；

2\)$$\eta_{1}, \eta_{2}, \cdots, \eta_{t}$$线性无关；

3\)$$Ax = 0$$的任一解都可以由$$\eta_{1}, \eta_{2}, \cdots, \eta_{t}$$线性表出$$k_{1} \eta_{1}+k_{2} \eta_{2}+\cdots+k_{t} \eta_{t}$$.是$$Ax = 0$$的通解，其中$$k_{1}, k_{2}, \cdots, k_{t}$$是任意常数。

# 矩阵的特征值和特征向量

## **1.矩阵的特征值和特征向量的概念及性质**

1. 设$$\lambda$$是Am的一个特征值，则$$kA,aA+bE,A^2,f(A),A^T,A^{-1},A^*$$有一个特征值分别为$$k\lambda,a\lambda+b,\lambda^2,\lambda^m,f(\lambda),\lambda,\lambda^{-1},\frac{|A|}{\lambda}$$且对应特征向量相同（$$A^T$$例外）。

   \(2\)若$$\lambda_1,\lambda_2,\dots,\lambda_n$$为A的n个特征值，则$$\sum_{i=1}^{n}\lambda_i=\sum_{i=1}^{n}a_{ii},\prod_{i=1}^{n}\lambda_i=|A|$$,从而$$|A| \neq 0 \Leftrightarrow A$$没有特征值。

   \(3\)设$$\lambda_1,\lambda_2,\dots,\lambda_s$$为A的s个特征值，对应特征向量为$$\alpha_{1}, \alpha_{2} \cdots \alpha_{s}$$，

   若:,$$\alpha=k_1\alpha_1+k_2\alpha_2+\dots+k_s\alpha_s$$

   则:$$A^n\alpha=k_1A^n\alpha_1+k_2A^n\alpha_2+\dots+k_sA^n\alpha_s=k_1\lambda_1^n\alpha_1+k_2\lambda_2^n\alpha_2+\dots+k_s\lambda_s^n\alpha_s$$。

## **2.相似变换、相似矩阵的概念及性质**

若$$A \sim B$$，则

1\)$$A^T \sim B^T, A^{-1} \sim B^{-1},A^* \sim B^*$$

2\)$$|A|=|B|, \sum_{i=1}^{n} A_{ii}=\sum_{i=1}^{n} B_{ii}, r(A)=r(B)$$

3\)$$|\lambda E-A|=|\lambda E-B|$$，对$$\forall \lambda$$成立

## **3.矩阵可相似对角化的充分必要条件**

\(1\)设A为n阶方阵，则A可对角化$$\Leftrightarrow$$对每个$$k_i$$重根特征值，有$$n-r\left(\lambda_{i} E-A\right)=k_{i}$$

\(2\) 设A可对角化，则由$$P^{-1} A P=\Lambda$$有$$A=P \Lambda P^{-1}$$，从而$$A^{n}=P \Lambda^{n} P^{-1}$$

\(3\) 重要结论

1\) 若$$A \sim B, C \sim D$$，则$$\left[ \begin{array}{ll}{A} & {O} \\ {O} & {C}\end{array}\right] \sim \left[ \begin{array}{ll}{B} & {O} \\ {O} & {D}\end{array}\right]$$.

2\) 若$$A \sim B$$，则$$f(A) \sim f(B)$$，$$|f(A)| \sim |f(B)|$$其中$$f(A)$$为关于n阶方阵A的多项式。

3\) 若A为可对角化矩阵，则其非零特征值的个数\(重根重复计算\)＝秩\(A\)

## **4.实对称矩阵的特征值、特征向量及相似对角阵**

\(1\)相似矩阵：设A,B为两个n阶方阵，如果存在一个可逆矩阵P，使得$$B=P^{-1} A P$$成立，则称矩阵A与B相似，记为$$A \sim B$$。

\(2\)相似矩阵的性质：如果则有$$A \sim B$$：

1\)$$A^{T} \sim B^{T}$$

2\)$$A^{-1} \sim B^{-1}$$（A若，B均可逆）

3\)$$A^{k} \sim B^{k}$$（k为正整数）

4\)$$|\lambda E-A|=|\lambda E-B|$$，从而A,B有相同的特征值

5\)$$|A|=|B|$$，从而A,B同时可逆或者不可逆

6\) 秩\(A\)=秩\(B\)，$$|\lambda E-A|=|\lambda E-B|$$, A,B不一定相似

# 二次型

## **1.n个变量**$$\mathbf{x}_{1}, \mathbf{x}_{2}, \cdots, \mathbf{x}_{\mathbf{n}}$$**的二次齐次函数**

$$f\left(x_{1}, x_{2}, \cdots, x_{n}\right)=\sum_{i=1}^{n} \sum_{j=1}^{n} a_{i j} x_{i} y_{j}$$，其中$$a_{i j}=a_{j i}(i, j=1,2, \cdots, n)$$，称为n元二次型，简称二次型. 若令$$x=\left[ \begin{array}{c}{x_{1}} \\ {x_{1}} \\ {\vdots} \\ {x_{n}}\end{array}\right], A=\left[ \begin{array}{cccc}{a_{11}} & {a_{12}} & {\cdots} & {a_{1 n}} \\ {a_{21}} & {a_{22}} & {\cdots} & {a_{2 n}} \\ {\cdots} & {\cdots} & {\cdots} & {\cdots} \\ {a_{n 1}} & {a_{n 2}} & {\cdots} & {a_{n n}}\end{array}\right]$$,这二次型f可改写成矩阵向量形式$$f=x^{T} A x$$。其中A称为二次型矩阵，因为$$a_{i j}=a_{j i}(i, j=1,2, \cdots, n)$$，所以二次型矩阵均为对称矩阵，且二次型与对称矩阵一一对应，并把矩阵A的秩称为二次型的秩。

## **2.惯性定理，二次型的标准形和规范形**

\(1\) 惯性定理

对于任一二次型，不论选取怎样的合同变换使它化为仅含平方项的标准型，其正负惯性指数与所选变换无关，这就是所谓的惯性定理。

\(2\) 标准形

二次型$$f=\left(x_{1}, x_{2}, \cdots, x_{n}\right)=x^{T} A x$$经过合同变换$$x =C y$$化为$$f=x^{T} A x=y^{T} C^{T} A C$$

$$y=\sum_{i=1}^{r} d_{i} y_{i}^{2}$$称为$$f(r \leq n)$$的标准形。在一般的数域内，二次型的标准形不是唯一的，与所作的合同变换有关，但系数不为零的平方项的个数由r\(A\)唯一确定。

\(3\) 规范形

任一实二次型f都可经过合同变换化为规范形$$f=z_{1}^{2}+z_{2}^{2}+\cdots z_{p}^{2}-z_{p+1}^{2}-\cdots-z_{r}^{2}$$，其中r为A的秩，p为正惯性指数，r-p为负惯性指数，且规范型唯一。

## **3.用正交变换和配方法化二次型为标准形，二次型及其矩阵的正定性**

设A正定$$\Rightarrow k A(k>0), A^{T}, A^{-1}, A^{*}$$正定；$$|A|>0$$,可逆；$$a_{ii} > 0$$，且$$\left|A_{i i}\right|>0$$

A,B正定$$\Rightarrow A+B$$正定，但AB，BA不一定正定

A正定$$\Leftrightarrow f(x)=x^{T} A x>0, \forall x \neq 0$$

$$\Leftrightarrow$$A的各阶顺序主子式全大于零

$$\Leftrightarrow$$A的所有特征值大于零

$$\Leftrightarrow$$A的正惯性指数为n

$$\Leftrightarrow$$存在可逆阵P使$$A=P^{T} P$$

$$\Leftrightarrow$$存在正交矩阵Q，使$$Q^{T} A Q=Q^{-1} A Q=\begin{pmatrix}
 \lambda_1&  & \\ 
 &  \dots& \\ 
 &  & \lambda_n
\end{pmatrix}$$

其中$$\lambda_{i}>0, i=1,2, \cdots, n$$正定$$\Rightarrow k A(k>0), A^{T}, A^{-1}, A^{*}$$正定；$$|A|>0,A$$可逆；$$a_{ii}>0$$，且$$|A_{ii}|>0$$。

