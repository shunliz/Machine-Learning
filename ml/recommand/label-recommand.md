1：联系用户兴趣和物品的方式

2：标签系统的典型代表

3：用户如何打标签

4：基于标签的推荐系统

5：算法的改进

6：标签推荐

# 一：联系用户兴趣和物品的方式

推荐系统的目的是联系用户的兴趣和物品，这种联系方式需要依赖不同的媒介。目前流行的推荐系统基本上是通过三种方式联系用户兴趣和物品。

![](/assets/label-recommand.png)

1：利用用户喜欢过的物品，给用户推荐与他喜欢过的物品相似的物品，即基于item的系统过滤推荐算法

2：利用用户和兴趣用户兴趣相似的其他用户，给用户推荐哪些和他们兴趣爱好相似的其他用户喜欢的物品，即基于User的协同过滤推荐算法

3：通过一些特征联系用户和物品，给用户推荐那些具有用户喜欢的特征的物品，这里的特征有不同的表现形式，比如可以表现为物品的属性集合，也可以表现为隐语义向量，而下面我们要讨论的是一种重要的特征表现形式——标签

# 基于标签的推荐系统

用户用标签来描述对物品的看法，因此标签是联系用户和物品的纽带，也是反应用户兴趣的重要数据源，如何利用用户的标签数据提高个性化推荐结果的质量是推荐系统研究的重要课题。

豆瓣很好地利用了标签数据，它将标签系统融入到了整个产品线中。

       首先，在每本书的页面上，豆瓣都提供了一个叫做“豆瓣成员常用标签”的应用，它给出了这本书上用户最常打的标签。

       同时，在用户给书做评价时，豆瓣也会让用户给图书打标签。

       最后，在最终的个性化推荐结果里，豆瓣利用标签将用户的推荐结果做了聚类，显示了对不同标签下用户的推荐结果，从而增加了推荐的多样性和可解释性。

        一个用户标签行为的数据集一般由一个三元组的集合表示，其中记录\(u, i, b\) 表示用户u给物品i打上了标签b。当然，用户的真实标签行为数据远远比三元组表示的要复杂，比如用户打标签的时间、用户的属性数据、物品的属性数据等。但是为了集中讨论标签数据，只考虑上面定义的三元组形式的数据，即用户的每一次打标签行为都用一个三元组（用户、物品、标签）表示。



## 1：试验设置

        本节将数据集随机分成10份。这里分割的键值是用户和物品，不包括标签。也就是说，用户对物品的多个标签记录要么都被分进训练集，要么都被分进测试集，不会一部分在训练集，另一部分在测试集中。然后，我们挑选1份作为测试集，剩下的9份作为训练集，通过学习训练集中的用户标签数据预测测试集上用户会给什么物品打标签。对于用户u，令R\(u\)为给用户u的长度为N的推荐列表，里面包含我们认为用户会打标签的物品。令T\(u\)是测试集中用户u实际上打过标签的物品集合。然后，我们利用准确率（precision）和召回率（recall）评测个性化推荐算法的精度。

![](http://img.blog.csdn.net/20160615201103047?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)![](http://img.blog.csdn.net/20160615201112456?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

         将上面的实验进行10次，每次选择不同的测试集，然后将每次实验的准确率和召回率的平均值作为最终的评测结果。为了全面评测个性化推荐的性能，我们同时评测了推荐结果的覆盖率（coverage）、多样性（diversity）和新颖度。覆盖率的计算公式如下：  


![](http://img.blog.csdn.net/20160615201158642?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

接下来我们用物品标签向量的余弦相似度度量物品之间的相似度。对于每个物品i，item\_tags\[i\]存储了物品i的标签向量，其中item\_tags\[i\]\[b\]是对物品i打标签b的次数，那么物品i和j的余弦相似度可以通过如下程序计算。

```py
#计算余弦相似度  
def CosineSim(item_tags,i,j):  
    ret = 0  
    for b,wib in item_tags[i].items():     #求物品i,j的标签交集数目  
        if b in item_tags[j]:  
            ret += wib * item_tags[j][b]  
    ni = 0  
    nj = 0  
    for b, w in item_tags[i].items():      #统计 i 的标签数目  
        ni += w * w  
    for b, w in item_tags[j].items():      #统计 j 的标签数目  
        nj += w * w  
    if ret == 0:  
        return 0  
    return ret/math.sqrt(ni * nj)          #返回余弦值  
```

在得到物品之间的相似度度量后，我们可以用如下公式计算一个推荐列表的多样性：



![](http://img.blog.csdn.net/20160615201914481?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)  


      Python实现为：

```py
#计算推荐列表多样性  
def Diversity(item_tags,recommend_items):  
    ret = 0  
    n = 0  
    for i in recommend_items.keys():  
        for j in recommend_items.keys():  
            if i == j:  
                continue  
            ret += CosineSim(item_tags,i,j)  
            n += 1  
    return ret/(n * 1.0)  
```



