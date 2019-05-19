# Numpy

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

    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]] [[[1 1 1 1]
      [1 1 1 1]
      [1 1 1 1]]
    
     [[1 1 1 1]
      [1 1 1 1]
      [1 1 1 1]]] [10 15 20] [ 0.    0.25  0.5   0.75  1.    1.25  1.5   1.75  2.  ] [[7 7]
     [7 7]] [[ 1.  0.]
     [ 0.  1.]] [[ 0.18387438  0.04048564]
     [ 0.68185975  0.02410229]] [[  1.39069238e-309   1.39069238e-309]
     [  1.39069238e-309   1.39069238e-309]
     [  1.39069238e-309   1.39069238e-309]]
    

输入输出
1，保存到磁盘和从磁盘导入
2，保存到文件和从文件导入


```python
np.save('my_array',a)
np.savez('array.npz',a,b)
np.load('my_array.npy')
```




    array([1, 2, 3])



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




    array([[1, 2, 3],
           [4, 5, 6]])



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




    numpy.unicode_



帮助


```python
np.info(np.ndarray.dtype)
```

    Data-type of the array's elements.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    d : numpy dtype object
    
    See Also
    --------
    numpy.dtype
    
    Examples
    --------
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.dtype
    dtype('int32')
    >>> type(x.dtype)
    <type 'numpy.dtype'>
    

数组运算
运算
比较
聚合


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

    [[ 1.5  2.   3. ]
     [ 4.   5.   6. ]]
    [[   4.48168907    7.3890561    20.08553692]
     [  54.59815003  148.4131591   403.42879349]]
    [[ 1.22474487  1.41421356  1.73205081]
     [ 2.          2.23606798  2.44948974]]
    [ 0.84147098  0.90929743  0.14112001]
    [[ 0.0707372  -0.41614684 -0.9899925 ]
     [-0.65364362  0.28366219  0.96017029]]
    [ 0.          0.69314718  1.09861229]
    [[ 7.  7.]
     [ 7.  7.]]
    


```python
a==b
print a<2
```

    [ True False False]
    


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

    [1 2 3]
    [[ 1.5  2.   3. ]
     [ 4.   5.   6. ]]
    6
    1
    [ 4.  5.  6.]
    [[  1.5   3.5   6.5]
     [  4.    9.   15. ]]
    2.0
    

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

    [1 2 3]
    [1 2 3]
    [1 2 3]
    [1 2 3]
    [1 3 3]
    [1 3 3]
    [2 3 3]
    

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

    []
    


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

    None
    [ 1.   3.   3.   0.   0.   0.   0.   0.   0.   0.   0.   0.  -0.5  0.   0.
     -3.  -3.  -3. ]
    [2 5 3 3]
    [2 3]
    


```python
print np.hsplit(a,3)
print np.vsplit(c,2)
```

    [array([2]), array([3]), array([3])]
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-30-b909db1843dc> in <module>()
          1 print np.hsplit(a,3)
    ----> 2 print np.vsplit(c,2)
    

    C:\ProgramData\Anaconda3\envs\py27\lib\site-packages\numpy\lib\shape_base.pyc in vsplit(ary, indices_or_sections)
        620     """
        621     if len(_nx.shape(ary)) < 2:
    --> 622         raise ValueError('vsplit only works on arrays of 2 or more dimensions')
        623     return split(ary, indices_or_sections, 0)
        624 
    

    ValueError: vsplit only works on arrays of 2 or more dimensions



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


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

# Scikit-learn

# pySpark

# Pandas

# tensorflow

# 



