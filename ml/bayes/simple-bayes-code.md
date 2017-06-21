```py
#encoding:utf-8

from numpy import *

#词表到向量的转换函数
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]      #1,侮辱  0,正常
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #调用set方法,创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)     #创建两个集合的并集
    return list(vocabSet)
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)   #创建一个所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word:%s is not in my Vocabulary" % word
    return returnVec
'''

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)   #创建一个所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#朴素贝叶斯分类器训练集
def trainNB0(trainMatrix,trainCategory):  #传入参数为文档矩阵，每篇文档类别标签所构成的向量
    numTrainDocs = len(trainMatrix)      #文档矩阵的长度
    numWords = len(trainMatrix[0])       #第一个文档的单词个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #任意文档属于侮辱性文档概率
    #p0Num = zeros(numWords);p1Num = zeros(numWords)        #初始化两个矩阵，长度为numWords，内容值为0
    p0Num = ones(numWords);p1Num = ones(numWords)        #初始化两个矩阵，长度为numWords，内容值为1
    #p0Denom = 0.0;p1Denom = 0.0                         #初始化概率
    p0Denom = 2.0;p1Denom = 2.0 
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num +=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num +=trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #p1Vect = p1Num/p1Denom #对每个元素做除法
    #p0Vect = p0Num/p0Denom
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   #元素相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()   #产生文档矩阵和对应的标签
    myVocabList = createVocabList(listOPosts) #创建并集
    trainMat = []   #创建一个空的列表
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))  #使用词向量来填充trainMat列表
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))  #训练函数
    testEntry = ['love','my','dalmation']   #测试文档列表
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry)) #声明矩阵
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))    #声明矩阵
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)
```



