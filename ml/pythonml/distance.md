# MachingLearning中的距离和相似性计算以及python实现

---

* # 欧氏距离 {#欧氏距离}

也称欧几里得距离，是指在m维空间中两个点之间的真实距离。欧式距离在ML中使用的范围比较广，也比较通用，就比如说利用k-Means对二维平面内的数据点进行聚类，对魔都房价的聚类分析（price/m^2 与平均房价）等。

两个n维向量a\($$x_{11},x_{12}.....x_{1n}$$\)与 b\($$x_{21},x_{22}.....x_{2n}$$\)间的欧氏距离

python 实现为：

```
def distance(a,b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])**2
    return sqrt(sum)

print 'a,b 多维距离为：',distance((1,1,2,2),(2,2,4,4))
```

这里传入的参数可以是任意维的，该公式也适应上边的二维和三维

* # 曼哈顿距离 {#曼哈顿距离}

$$D_{12}=\sum_{k=1}^{n}|x_{1k}-x_{2k}|$$

python 实现为：

```
def threeMHDdis(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1]) + abs(a[2]-b[2])

print 'a,b 三维曼哈顿距离为：', threeMHDdis((1,1,1),(2,2,2))
```

* # 切比雪夫距离 {#切比雪夫距离}

切比雪夫距离（Chebyshev Distance）的定义为：max\( \| x2-x1 \| , \|y2-y1 \| , … \), 切比雪夫距离用的时候数据的维度必须是三个以上

python 实现为：

```
def moreQBXFdis(a,b):
    maxnum = 0
    for i in range(len(a)):
        if abs(a[i]-b[i]) > maxnum:
            maxnum = abs(a[i]-b[i])
    return maxnum

print 'a,b多维切比雪夫距离：' , moreQBXFdis((1,1,1,1),(3,4,3,4))
```

* # 马氏距离 {#马氏距离}

有M个样本向量X1~Xm，协方差矩阵记为S，均值记为向量μ，则其中样本向量X到u的马氏距离表示为

$$D(x)=\sqrt {(X-u)^TS^{-1}(X-u)}$$

* # 夹角余弦 {#夹角余弦}

$$cos \theta = \frac {a*b} {|a||b|}$$              $$cos \theta = \frac {\sum_{k=1}^{n}x_{1k}x_{2k}} {\sqrt {\sum_{k=1}^{n}x_{1k}^2}\sqrt {\sum_{k=1}^{n}x_{2k}^2}}$$

```
def moreCos(a,b):
    sum_fenzi = 0.0
    sum_fenmu_1,sum_fenmu_2 = 0,0
    for i in range(len(a)):
        sum_fenzi += a[i]*b[i]
        sum_fenmu_1 += a[i]**2 
        sum_fenmu_2 += b[i]**2 

    return sum_fenzi/( sqrt(sum_fenmu_1) * sqrt(sum_fenmu_2) )
print 'a,b 多维夹角余弦距离：',moreCos((1,1,1,1),(2,2,2,2))
```



