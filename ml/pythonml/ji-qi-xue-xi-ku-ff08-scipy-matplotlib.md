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

# matplotlib

---

```python
import matplotlib.pyplot as plt
x = [1,2,3,4]
y = [10,20,25,30]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, color='lightblue', linewidth=3)
ax.scatter([2,4,6],[5,15,25],color='darkgreen',marker='^')
ax.set_xlim(1, 6.5)
plt.savefig('foo.png')
plt.show()
```

准备数据

```python
import numpy as np
x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)
```

```python
data = 2 * np.random.random((10, 10))
data2 = 3 * np.random.random((10, 10))
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
from matplotlib.cbook import get_sample_data
img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))
```

画图

```python
import matplotlib.pyplot as plt
fig = plt.figure()
fig2 = plt.figure(figsize=plt.figaspect(2.0))

fig.add_axes()
ax1 = fig.add_subplot(221) # row-col-num
ax3 = fig.add_subplot(212)
fig3, axes = plt.subplots(nrows=2,ncols=2)
fig4, axes2 = plt.subplots(ncols=3)
plt.show()
```

定制画布

```python
#颜色
plt.plot(x, x, x, x**2, x, x**3)
plt.show()
ax.plot(x, y, alpha = 0.4)
plt.show()
ax.plot(x, y, c='k')
plt.show()
#fig.colorbar(im, orientation='horizontal')
#im = ax.imshow(img,cmap='seismic')
#标记
fig, ax = plt.subplots()
ax.scatter(x,y,marker=".")
ax.plot(x,y,marker="o")
#线型
plt.plot(x,y,linewidth=4.0)
plt.plot(x,y,ls='solid')
plt.plot(x,y,ls='--')
plt.plot(x,y,'--',x**2,y**2,'-.')
plt.setp(lines,color='r',linewidth=4.0)
#文字装饰
ax.text(1,-2.1, 'Example Graph',style='italic')
ax.annotate("Sine", xy=(8, 0),xycoords='data',xytext=(10.5, 0),textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),)
#数学公式
plt.title(r'$sigma_i=15$', fontsize=20)
#标题和布局
ax.margins(x=0.0,y=0.1)
ax.axis('equal')
ax.set(xlim=[0,10.5],ylim=[-1.5,1.5])
ax.set_xlim(0,10.5)

ax.set(title='An Example Axes',ylabel='Y-Axis',xlabel='X-Axis')
ax.legend(loc='best')

ax.xaxis.set(ticks=range(1,5),ticklabels=[3,100,-12,"foo"],direction='inout',length=10)

fig3.subplots_adjust(wspace=0.5,hspace=0.3,left=0.125,right=0.9,top=0.9,bottom=0.1)
fig.tight_layout()

ax1.spines['top'=].set_visible(False)
ax1.spines['bottom'].set_position(('outward',10))
```

画图函数

```python
#1D
lines = ax.plot(x,y)
ax.scatter(x,y)
axes[0,0].bar([1,2,3],[3,4,5])
axes[1,0].barh([0.5,1,2.5],[0,1,2])
axes[1,1].axhline(0.45)
axes[0,1].axvline(0.65)
ax.fill(x,y,color='blue')
ax.fill_between(x,y,color='yellow')
#2D
fig, ax = plt.subplots()
im = ax.imshow(img,arrays cmap='gist_earth',interpolation='nearest',vmin=-2,vmax=2)

#向量
axes[0,1].arrow(0,0,0.5,0.5)
axes[1,1].quiver(y,z)
axes[0,1].streamplot(X,Y,U,V)

#数据分布
ax1.hist(y)
ax3.boxplot(y)
ax3.violinplot(z)

axes2[0].pcolor(data2)
axes2[0].pcolormesh(data)
CS = plt.contour(Y,X,U)
axes2[2].contourf(data1)
axes2[2]= ax.clabel(CS)
```

```python
plt.show()
```

保存画布

```python
plt.savefig('foo.png')
plt.savefig('foo.png', transparent=True)
```

关闭清除

```python
plt.cla()
plt.clf()
plt.close()
```



