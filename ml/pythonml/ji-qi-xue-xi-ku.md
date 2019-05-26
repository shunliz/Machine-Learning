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

---

基本的scikit-learn操作代码

```python
from sklearn import neighbors, datasets, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=33)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
```

Normalization这个名词在很多地方都会出现，但是对于数据却有两种截然不同且容易混淆的处理过程。对于某个多特征的机器学习数据集来说，第一种Normalization是对于将数据进行预处理时进行的操作，是对于数据集的各个特征分别进行处理，主要包括min-max normalization、Z-score normalization、 log函数转换和atan函数转换等。第二种Normalization对于每个样本缩放到单位范数（每个样本的范数为1），主要有L1-normalization（L1范数）、L2-normalization（L2范数）等，可以用于SVM等应用

第一种 Normalization  
数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间。在某些比较和评价的指标处理中经常会用到，去除数据的单位限制，将其转化为无量纲的纯数值，便于不同单位或量级的指标能够进行比较和加权。其中最典型的就是数据的标准化处理，即将数据统一映射到\[0,1\]区间上。标准化在0-1之间是统计的概率分布，标准化在某个区间上是统计的坐标分布。目前数据标准化方法有多种。不同的标准化方法，对系统的评价结果会产生不同的影响，然而不幸的是，在数据标准化方法的选择上，还没有通用的法则可以遵循。

标准化（normalization）的目的：  
在数据分析之前，我们通常需要先将数据标准化（normalization），利用标准化后的数据进行数据分析。数据标准化处理主要包括数据同趋化处理和无量纲化处理两个方面。数据同趋化处理主要解决不同性质数据问题，对不同性质指标直接加总不能正确反映不同作用力的综合结果，须先考虑改变逆指标数据性质，使所有指标对测评方案的作用力同趋化，再加总才能得出正确结果。数据无量纲化处理主要解决数据的可比性。经过上述标准化处理，原始数据均转换为无量纲化指标测评值，即各指标值都处于同一个数量级别上，可以进行综合测评分析。也就说标准化（normalization）的目的是：

把特征的各个维度标准化到特定的区间

把有量纲表达式变为无量纲表达式

归一化后有两个好处：  
1. 加快基于梯度下降法或随机梯度下降法模型的收敛速度  
2. 提升模型的精度

数据预处理：标准化 $$x={x-u}/\delta$$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
```

```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
```

二值化

```python
from sklearn.preprocessing import Binarizer
X = [[ 1., -1.,  2.],
    [ 2.,  0.,  0.],
    [ 0.,  1., -1.]]
binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
binarizer.transform(X)
```

特征类别编码

模型初始化

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
from sklearn.svm import SVC
svc = SVC(kernel='linear')
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
```

模型训练

```python
lr.fit(X, y)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
k_means.fit(X_train)
#pca_model = pca.fit_transform(X_train)
```

预测

```python
y_pred = svc.predict(np.random.radom((2,5)))
y_pred = lr.predict(X_test)
y_pred = knn.predict_proba(X_test)
```

数据预处理

```python
#Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)

#Normalization
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)

#Binarization
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)

#Encoding Categorical Features
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit_transform(X_train)

#Imputing Missing Values
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit_transform(X_train)

#Generating Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(5)
poly.fit_transform(X)
```

评价你的模型：分类算法的指标

```python
#Accuracy Score
knn.score(X_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
```

评价你的模型：回归算法的指标

```python
#Mean Absolute Error
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2]
mean_absolute_error(y_true, y_pred)
#Mean Squared Error
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
#R² Score
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)
```

评价你的模型：聚类算法的指标

```python
#Adjusted Rand Index
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_true, y_pred)
#Homogeneity
from sklearn.metrics import homogeneity_score
homogeneity_score(y_true, y_pred)
#V-measure
from sklearn.metrics import v_measure_score
metrics.v_measure_score(y_true, y_pred)
```

评价你的模型：交叉验证

```python
#Cross-Validation
from sklearn.cross_validation import cross_val_score
print(cross_val_score(knn, X_train, y_train, cv=4))
print(cross_val_score(lr, X, y, cv=2))
```

训练模型

```python
#监督学习
lr.fit(X, y)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
#无监督学习
k_means.fit(X_train)
pca_model = pca.fit_transform(X_train)
```

训练和测试

```python
from sklearn.cross validation import train_test_split
X train, X test, y train, y test - train_test_split(X,y,random state-0)
```

模型调优

```python
#Grid Search
from sklearn.grid_search import GridSearchCV
params = {"n_neighbors": np.arange(1,3),"metric": ["euclidean","cityblock"]}
grid = GridSearchCV(estimator=knn,param_grid=params)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)

#Randomized Parameter Optimization
from sklearn.grid_search import RandomizedSearchCV
params = {"n_neighbors": range(1,5),"weights": ["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn,param_distributions=params,cv=4,n_iter=8,random_state=5)
rsearch.fit(X_train, y_train)
print(rsearch.best_score_)
```

# Scipy

---


```python
import numpy as np
a = np.array([1,2,3])
b = np.array([(1+5j,2j,3j), (4j,5j,6j)])
c = np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]])
```


```python
from scipy import linalg, sparse
```


```python
#Creating Matrices
A = np.matrix(np.random.random((2,2)))
B = np.asmatrix(b)
C = np.mat(np.random.random((10,5)))
D = np.mat([[3,4], [5,6]])
print A
print B
print C
print D
```


```python
A.I  #inverse
linalg.inv(A)
```


```python
A.T   #transpose
A.H 
```


```python
np.trace(A)
```


```python
print linalg.norm(A)
print linalg.norm
print linalg.norm(A,np.inf)
```


```python
np.linalg.matrix_rank(C)
```


```python
linalg.det(A)
```


```python
linalg.pinv(C)
```


```python
linalg.pinv2(C)
```


```python
np.add(A,D)
np.subtract(A,D)
np.divide(A,D)
np.multiply(D,A)
np.dot(A,D)
np.vdot(A,D)
np.inner(A,D)
np.outer(A,D)
np.tensordot(A,D)
np.kron(A,D)
linalg.expm(A)
linalg.logm(A)
linalg.sinm(D)
linalg.cosm(D)
linalg.tanm(A)
linalg.sinhm(D)
linalg.coshm(D)
linalg.tanhm(A)
np.signm(A)
linalg.sqrtm(A)
linalg.funm(A, lambda x: x*x)
```

矩阵分解


```python
#eigen
la, v = linalg.eig(A)
l1, l2 = la
v[:,0]
v[:,1]
linalg.eigvals(A)

#SVD
U,s,Vh = linalg.svd(B)
M,N = B.shape
Sig = linalg.diagsvd(s,M,N)

#LU
P,L,U = linalg.lu(C)
print P
print L
print U
```

