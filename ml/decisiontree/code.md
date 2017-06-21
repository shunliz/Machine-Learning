```py
#coding=utf-8  
''''' 
'''  
from math import log  
import operator  
  
def createDataSet():  
    dataSet =[[1,1,'yes'],  
              [1,1,'yes'],  
              [1,0,'no'],  
              [0,1,'no'],  
              [0,1,'no']]  
    labels = ['no surfacing','flippers'] #分类的属性  
    return dataSet,labels  
  
#计算给定数据的香农熵  
def calcShannonEnt(dataSet):  
    numEntries = len(dataSet)  
    labelCounts = {}  
    for featVec in dataSet:  
        currentLabel = featVec[-1] #获得标签  
        #构造存放标签的字典  
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel]=0  
        labelCounts[currentLabel]+=1 #对应的标签数目+1  
    #计算香农熵  
    shannonEnt = 0.0  
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries  
        shannonEnt -=prob*log(prob,2)  
    return shannonEnt  
  
#划分数据集,三个参数为带划分的数据集，划分数据集的特征，特征的返回值  
def splitDataSet(dataSet,axis,value):    
    retDataSet = []  
    for featVec in dataSet:  
        if featVec[axis] ==value:  
            #将相同数据集特征的抽取出来  
            reducedFeatVec = featVec[:axis]  
            reducedFeatVec.extend(featVec[axis+1:])  
            retDataSet.append(reducedFeatVec)  
    return retDataSet #返回一个列表  
          
#选择最好的数据集划分方式  
def chooseBestFeatureToSplit(dataSet):  
    numFeature = len(dataSet[0])-1  
    baseEntropy = calcShannonEnt(dataSet)  
    bestInfoGain = 0.0  
    beatFeature = -1  
    for i in range(numFeature):  
        featureList = [example[i] for example in dataSet] #获取第i个特征所有的可能取值  
        uniqueVals = set(featureList)  #从列表中创建集合，得到不重复的所有可能取值ֵ  
        newEntropy = 0.0  
        for value in uniqueVals:  
            subDataSet = splitDataSet(dataSet,i,value)   #以i为数据集特征，value为返回值，划分数据集  
            prob = len(subDataSet)/float(len(dataSet))   #数据集特征为i的所占的比例  
            newEntropy +=prob * calcShannonEnt(subDataSet)   #计算每种数据集的信息熵  
        infoGain = baseEntropy- newEntropy  
        #计算最好的信息增益，增益越大说明所占决策权越大  
        if (infoGain > bestInfoGain):  
            bestInfoGain = infoGain  
            bestFeature = i  
    return bestFeature  
  
#递归构建决策树  
def majorityCnt(classList):        
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys():  
            classCount[vote]=0  
        classCount[vote]+=1  
    sortedClassCount = sorted(classCount.iteritems(),key =operator.itemgetter(1),reverse=True)#排序，True升序  
    return sortedClassCount[0][0]  #返回出现次数最多的  
  
 #创建树的函数代码  
def createTree(dataSet,labels):       
    classList = [example[-1]  for example in dataSet]
    if classList.count(classList[0])==len(classList):	
    #if set(classList)==set([classList[0]]):#类别完全相同则停止划分  
        return classList[0]  
    if len(dataSet[0]) ==1:             #遍历完所有特征值时返回出现次数最多的  
        return majorityCnt(classList)  
    bestFeat = chooseBestFeatureToSplit(dataSet)   #选择最好的数据集划分方式  
    bestFeatLabel = labels[bestFeat]   #得到对应的标签值  
    myTree = {bestFeatLabel:{}}  
    del(labels[bestFeat])      #清空labels[bestFeat],在下一次使用时清零  
    featValues = [example[bestFeat] for example in dataSet]   
    uniqueVals = set(featValues)  
    for value in uniqueVals:  
        subLabels =labels[:]  
        #递归调用创建决策树函数  
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)  
    return myTree    
  
if __name__=="__main__":  
    dataSet,labels = createDataSet()  
    print createTree(dataSet,labels) 

```



