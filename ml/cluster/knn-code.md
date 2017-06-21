```py
#encoding:utf-8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    #返回“数组”的行数，如果shape[1]返回的则是数组的列数
    dataSetSize = dataSet.shape[0]
    #两个“数组”相减，得到新的数组
    diffMat = tile(inX,(dataSetSize,1))- dataSet
    #求平方
    sqDiffMat = diffMat **2
    #求和，返回的是一维数组
    sqDistances = sqDiffMat.sum(axis=1)
    #开方，即测试点到其余各个点的距离
    distances = sqDistances **0.5
    #排序，返回值是原数组从小到大排序的下标值
    sortedDistIndicies = distances.argsort()
    #定义一个空的字典
    classCount = {}
    for i in range(k):
        #返回距离最近的k个点所对应的标签值
        voteIlabel = labels[sortedDistIndicies[i]]
        #存放到字典中
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #排序 classCount.iteritems() 输出键值对 key代表排序的关键字 True代表降序
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    #返回距离最小的点对应的标签
   return sortedClassCount[0][0]
```



