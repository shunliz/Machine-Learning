# 中文文本挖掘预处理

---

在对文本做数据分析时，我们一大半的时间都会花在文本预处理上，而中文和英文的预处理流程稍有不同，本文就对中文文本挖掘的预处理流程做一个总结。

## 1. 中文文本挖掘预处理特点

首先我们看看中文文本挖掘预处理和英文文本挖掘预处理相比的一些特殊点。

首先，中文文本是没有像英文的单词空格那样隔开的，因此不能直接像英文一样可以直接用最简单的空格和标点符号完成分词。所以一般我们需要用分词算法来完成分词，在[文本挖掘的分词原理](/nlp/text-mine.md)中，我们已经讲到了中文的分词原理，这里就不多说。

第二，中文的编码不是utf8，而是unicode。这样会导致在分词的时候，和英文相比，我们要处理编码的问题。

这两点构成了中文分词相比英文分词的一些不同点，后面我们也会重点讲述这部分的处理。当然，英文分词也有自己的烦恼，这个我们在以后再讲。了解了中文预处理的一些特点后，我们就言归正传，通过实践总结下中文文本挖掘预处理流程。

## 2.  中文文本挖掘预处理一：数据收集

在文本挖掘之前，我们需要得到文本数据，文本数据的获取方法一般有两种：使用别人做好的语料库和自己用爬虫去在网上去爬自己的语料数据。

对于第一种方法，常用的文本语料库在网上有很多，如果大家只是学习，则可以直接下载下来使用，但如果是某些特殊主题的语料库，比如“机器学习”相关的语料库，则这种方法行不通，需要我们自己用第二种方法去获取。

对于第二种使用爬虫的方法，开源工具有很多，通用的爬虫我一般使用[beautifulsoup](http://link.zhihu.com/?target=http%3A//www.crummy.com/software/BeautifulSoup/)。但是我们我们需要某些特殊的语料数据，比如上面提到的“机器学习”相关的语料库，则需要用主题爬虫（也叫聚焦爬虫）来完成。这个我一般使用[ache](https://github.com/ViDA-NYU/ache)。 ache允许我们用关键字或者一个分类算法来过滤出我们需要的主题语料，比较强大。

## 3.  中文文本挖掘预处理二：除去数据中非文本部分

这一步主要是针对我们用爬虫收集的语料数据，由于爬下来的内容中有很多html的一些标签，需要去掉。少量的非文本内容的可以直接用Python的正则表达式\(re\)删除, 复杂的则可以用[beautifulsoup](http://link.zhihu.com/?target=http%3A//www.crummy.com/software/BeautifulSoup/)来去除。去除掉这些非文本的内容后，我们就可以进行真正的文本预处理了。

## 4. 中文文本挖掘预处理三：处理中文编码问题

由于Python2不支持unicode的处理，因此我们使用Python2做中文文本预处理时需要遵循的原则是，存储数据都用utf8，读出来进行中文相关处理时，使用GBK之类的中文编码，在下面一节的分词时，我们再用例子说明这个问题。

## 5. 中文文本挖掘预处理四：中文分词

常用的中文分词软件有很多，个人比较推荐结巴分词。安装也很简单，比如基于Python的，用"pip install jieba"就可以完成。下面我们就用例子来看看如何中文分词。

首先我们准备了两段文本，这两段文本在两个文件中。两段文本的内容分别是nlp\_test0.txt和nlp\_test2.txt：

> 沙瑞金赞叹易学习的胸怀，是金山的百姓有福，可是这件事对李达康的触动很大。易学习又回忆起他们三人分开的前一晚，大家一起喝酒话别，易学习被降职到道口县当县长，王大路下海经商，李达康连连赔礼道歉，觉得对不起大家，他最对不起的是王大路，就和易学习一起给王大路凑了5万块钱，王大路自己东挪西撮了5万块，开始下海经商。没想到后来王大路竟然做得风生水起。沙瑞金觉得他们三人，在困难时期还能以沫相助，很不容易。
>
> 沙瑞金向毛娅打听他们家在京州的别墅，毛娅笑着说，王大路事业有成之后，要给欧阳菁和她公司的股权，她们没有要，王大路就在京州帝豪园买了三套别墅，可是李达康和易学习都不要，这些房子都在王大路的名下，欧阳菁好像去住过，毛娅不想去，她觉得房子太大很浪费，自己家住得就很踏实。  
> \`\`\`

我们先讲文本从第一个文件中读取，并使用中文GBK编码，再调用结巴分词，最后把分词结果用uft8格式存在另一个文本nlp\_test1.txt

中。代码如下：

```py
# -*- coding: utf-8 -*-

import jieba

with open('./nlp_test0.txt') as f:
    document = f.read()

    document_decode = document.decode('GBK')
    document_cut = jieba.cut(document_decode)
    #print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('./nlp_test1.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()
```

输出的文本内容如下：

> 沙 瑞金 赞叹 易 学习 的 胸怀 ， 是 金山 的 百姓 有福 ， 可是 这件 事对 李达康 的 触动 很大 。 易 学习 又 回忆起 他们 三人 分开 的 前一晚 ， 大家 一起 喝酒 话别 ， 易 学习 被 降职 到 道口 县当 县长 ， 王 大路 下海经商 ， 李达康 连连 赔礼道歉 ， 觉得 对不起 大家 ， 他 最 对不起 的 是 王 大路 ， 就 和 易 学习 一起 给 王 大路 凑 了 5 万块 钱 ， 王 大路 自己 东挪西撮 了 5 万块 ， 开始 下海经商 。 没想到 后来 王 大路 竟然 做 得 风生水 起 。 沙 瑞金 觉得 他们 三人 ， 在 困难 时期 还 能 以沫 相助 ， 很 不 容易 。

可以发现对于一些人名和地名，jieba处理的不好，不过我们可以帮jieba加入词汇如下：

```
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)
```

现在我们再来进行读文件，编码，分词，编码和写文件，代码如下：

```py
with open('./nlp_test0.txt') as f:
    document = f.read()

    document_decode = document.decode('GBK')
    document_cut = jieba.cut(document_decode)
    #print  ' '.join(jieba_cut)
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('./nlp_test1.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()
```

输出的文本内容如下：

> 沙瑞金 赞叹 易学习 的 胸怀 ， 是 金山 的 百姓 有福 ， 可是 这件 事对 李达康 的 触动 很大 。 易学习 又 回忆起 他们 三人 分开 的 前一晚 ， 大家 一起 喝酒 话别 ， 易学习 被 降职 到 道口 县当 县长 ， 王大路 下海经商 ， 李达康 连连 赔礼道歉 ， 觉得 对不起 大家 ， 他 最 对不起 的 是 王大路 ， 就 和 易学习 一起 给 王大路 凑 了 5 万块 钱 ， 王大路 自己 东挪西撮 了 5 万块 ， 开始 下海经商 。 没想到 后来 王大路 竟然 做 得 风生水 起 。 沙瑞金 觉得 他们 三人 ， 在 困难 时期 还 能 以沫 相助 ， 很 不 容易 。

基本已经可以满足要求。同样的方法我们对第二段文本nlp\_test2.txt进行分词和写入文件nlp\_test3.txt。

```py
with open('./nlp_test2.txt') as f:
    document2 = f.read()

    document2_decode = document2.decode('GBK')
    document2_cut = jieba.cut(document2_decode)
    #print  ' '.join(jieba_cut)
    result = ' '.join(document2_cut)
    result = result.encode('utf-8')
    with open('./nlp_test3.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()
```

输出的文本内容如下：

> 沙瑞金 向 毛娅 打听 他们 家 在 京州 的 别墅 ， 毛娅 笑 着 说 ， 王大路 事业有成 之后 ， 要 给 欧阳 菁 和 她 公司 的 股权 ， 她们 没有 要 ， 王大路 就 在 京州 帝豪园 买 了 三套 别墅 ， 可是 李达康 和 易学习 都 不要 ， 这些 房子 都 在 王大路 的 名下 ， 欧阳 菁 好像 去 住 过 ， 毛娅 不想 去 ， 她 觉得 房子 太大 很 浪费 ， 自己 家住 得 就 很 踏实 。

可见分词效果还不错。

## 6. 中文文本挖掘预处理五：引入停用词

在上面我们解析的文本中有很多无效的词，比如“着”，“和”，还有一些标点符号，这些我们不想在文本分析的时候引入，因此需要去掉，这些词就是停用词。常用的中文停用词表是1208个，[下载地址在这](http://files.cnblogs.com/files/pinard/stop_words.zip)。当然也有其他版本的停用词表，不过这个1208词版是我常用的。

在我们用scikit-learn做特征处理的时候，可以通过参数stop\_words来引入一个数组作为停用词表。

现在我们将停用词表从文件读出，并切分成一个数组备用：

```
#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list  
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()
```

## 7. 中文文本挖掘预处理六：特征处理

现在我们就可以用scikit-learn来对我们的文本特征进行处理了，在[文本挖掘预处理之向量化与Hash Trick](http://www.cnblogs.com/pinard/p/6688348.html)中，我们讲到了两种特征处理的方法，向量化与Hash Trick。而向量化是最常用的方法，因为它可以接着进行TF-IDF的特征处理。在[文本挖掘预处理之TF-IDF](http://www.cnblogs.com/pinard/p/6693230.html)中，我们也讲到了TF-IDF特征处理的方法。这里我们就用scikit-learn的TfidfVectorizer类来进行TF-IDF特征处理。

TfidfVectorizer类可以帮助我们完成向量化，TF-IDF和标准化三步。当然，还可以帮我们处理停用词。

现在我们把上面分词好的文本载入内存：

```
with open('./nlp_test1.txt') as f3:
    res1 = f3.read()
print res1
with open('./nlp_test3.txt') as f4:
    res2 = f4.read()
print res2
```

这里的输出还是我们上面分完词的文本。现在我们可以进行向量化，TF-IDF和标准化三步处理了。注意，这里我们引入了我们上面的停用词表。

```
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [res1,res2]
vector = TfidfVectorizer(stop_words=stpwrdlst)
tfidf = vector.fit_transform(corpus)
print tfidf
```

部分输出如下：

```
  (0, 44)    0.154467434933
  (0, 59)    0.108549295069
  (0, 39)    0.308934869866
  (0, 53)    0.108549295069
　　....
  (1, 27)    0.139891059658
  (1, 47)    0.139891059658
  (1, 30)    0.139891059658
  (1, 60)    0.139891059658
```

我们再来看看每次词和TF-IDF的对应关系：

```
wordlist = vector.get_feature_names()#获取词袋模型中的所有词  
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()  
#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):  
    print "-------第",i,"段文本的词语tf-idf权重------"  
    for j in range(len(wordlist)):  
        print wordlist[j],weightlist[i][j]
```

部分输出如下：

```
-------第 0 段文本的词语tf-idf权重------
一起 0.217098590137
万块 0.217098590137
三人 0.217098590137
三套 0.0
下海经商 0.217098590137
.....
-------第 1 段文本的词语tf-idf权重------
.....
李达康 0.0995336411066
欧阳 0.279782119316
毛娅 0.419673178975
沙瑞金 0.0995336411066
没想到 0.0
没有 0.139891059658
浪费 0.139891059658
王大路 0.29860092332
.....
```

## 8. 中文文本挖掘预处理七：建立分析模型

有了每段文本的TF-IDF的特征向量，我们就可以利用这些数据建立分类模型，或者聚类模型了，或者进行主题模型的分析。比如我们上面的两段文本，就可以是两个训练样本了。此时的分类聚类模型和之前讲的非自然语言处理的数据分析没有什么两样。因此对应的算法都可以直接使用。而主题模型是自然语言处理比较特殊的一块，这个我们后面再单独讲。

## 9.中文文本挖掘预处理总结

上面我们对中文文本挖掘预处理的过程做了一个总结，希望可以帮助到大家。需要注意的是这个流程主要针对一些常用的文本挖掘，并使用了词袋模型，对于某一些自然语言处理的需求则流程需要修改。比如我们涉及到词上下文关系的一些需求，此时不能使用词袋模型。而有时候我们对于特征的处理有自己的特殊需求，因此这个流程仅供自然语言处理入门者参考。

