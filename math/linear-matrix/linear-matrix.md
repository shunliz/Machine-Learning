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
3. 


