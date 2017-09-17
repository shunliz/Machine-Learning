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

python 实现为：s

```
def threeMHDdis(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1]) + abs(a[2]-b[2])

print 'a,b 三维曼哈顿距离为：', threeMHDdis((1,1,1),(2,2,2)) 
```



