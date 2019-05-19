# Numpy

---

创建数组

```python
import numpy as np

a = np.array([1,2,3])
b = np.array([(1.5,2,3), (4,5,6)], dtype = float)
c = np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]],dtype = float)
```

创建占位符

```python
z1=np.zeros((3,4))
z2=np.ones((2,3,4),dtype=np.int16)
z3=d= np.arange(10,25,5)
z4=np.linspace(0,2,9)
z5 =e= np.full((2,2),7)
z6 =f= np.eye(2)
z7=np.random.random((2,2))
z8=np.empty((3,2))
print z1,z2,z3,z4,z5,z6,z7,z8
```

输入输出  
1，保存到磁盘和从磁盘导入  
2，保存到文件和从文件导入

```python
np.save('my_array',a)
np.savez('array.npz',a,b)
np.load('my_array.npy')
```

查看数组信息

```python
a.shape
len(a)
b.ndim
z1.size
b.dtype
b.dtype.name
b.astype(int)
```

数据类型

```python
np.int64
np.float32
np.complex
np.bool
np.object
np.string_
np.unicode_
```

帮助

```python
np.info(np.ndarray.dtype)
```

数组运算:1, 运算 2, 比较 3,聚合

```python
g=a-b
np.subtract(a,b)
b+a
np.add(b,a)
a/b
np.divide(a,b)
a*b
np.multiply(a,b)
print b
print np.exp(b)
print  np.sqrt(b)
print  np.sin(a)
print  np.cos(b)
print  np.log(a)
print  e.dot(f)
```

```python
a==b
print a<2
```

```python
print a
print b
print a.sum()
print a.min()
print b.max(axis=0)
print b.cumsum(axis=1)
print a.mean()
#print b.median()
```

拷贝数组

```python
print a
h = a.view()
print h
c =np.copy(a)
print c
a[1]=3
print c
h = a.copy()
print h
a[0] =2
print h
print a
```

数组切片，布尔索引，高级索引

```python
print a
print b
print c
print a[0:2]   
print b[0:2,1]
print b[:1]
print c[1,...]
print a[::-1]
```

```python
print a[a<2]    #选取所有a<2的元素
```

```python
b[[1,0,1,0],[0,1,2,0]] #选取(1,0),(0,1),(1,2) 和(0,0)
```

数组操作;1，转置。 2，增加删除元素。 3，切分数组。4，改变数组形状。5，合并数组

```python
i=np.transpose(b)
```

```python
print h.resize(2,6)
print np.append(h,g)
print np.insert(a,1,5)
print np.delete(a,[1])
```

```python
print np.hsplit(a,3)
print np.vsplit(c,2)
```

```python
b.ravel() #数组扁平化
g.reshape(3,-2) #
```

```python
print np.concatnate((a,d),axis=0)
print np.vstack((a,b))
print np.r_[e,f]
print np.hstack((e,f))
print np.column_stack((a,d))
print np.c_[a,d]
```

# Scikit-learn

# pySpark

# Pandas

# tensorflow

# 



