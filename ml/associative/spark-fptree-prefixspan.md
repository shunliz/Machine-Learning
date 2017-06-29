# 用Spark学习FP Tree算法和PrefixSpan算法

---

　在[FP Tree算法原理总结](/ml/associative/fptree.md)和[PrefixSpan算法原理总结](/ml/associative/prefixspan.md)中，我们对FP Tree和PrefixSpan这两种关联算法的原理做了总结，这里就从实践的角度介绍如何使用这两个算法。由于scikit-learn中没有关联算法的类库，而Spark MLlib有，本文的使用以Spark MLlib作为使用环境。

# 1. Spark MLlib关联算法概述

　　　　在Spark MLlib中，也只实现了两种关联算法，即我们的FP Tree和PrefixSpan，而像Apriori,GSP之类的关联算法是没有的。而这些算法支持Python,Java,Scala和R的接口。由于前面的实践篇我们都是基于Python，本文的后面的介绍和使用也会使用MLlib的Python接口。

　　　　 Spark MLlib关联算法基于Python的接口在pyspark.mllib.fpm包中。FP Tree算法对应的类是pyspark.mllib.fpm.FPGrowth\(以下简称FPGrowth类\)，从Spark1.4开始才有。而PrefixSpan算法对应的类是pyspark.mllib.fpm.PrefixSpan\(以下简称PrefixSpan类\)，从Spark1.6开始才有。因此如果你的学习环境的Spark低于1.6的话，是不能正常的运行下面的例子的。

　　　　 Spark MLlib也提供了读取关联算法训练模型的类，分别是 pyspark.mllib.fpm.FPGrowthModel和pyspark.mllib.fpm.PrefixSpanModel。这两个类可以把我们之前保存的FP Tree和PrefixSpan训练模型读出来。

# 2. Spark MLlib关联算法参数介绍

　　　　对于FPGrowth类，使用它的训练函数train主要需要输入三个参数：数据项集data，支持度阈值minSupport和数据并行运行时的数据分块数numPartitions。对于支持度阈值minSupport，它的取值大小影响最后的频繁项集的集合大小，支持度阈值越大，则最后的频繁项集数目越少，默认值0.3。而数据并行运行时的数据分块数numPartitions主要在分布式环境的时候有用，如果你是单机Spark，则可以忽略这个参数。

　　　　对于PrefixSpan类， 使用它的训练函数train主要需要输入四个参数：序列项集data，支持度阈值minSupport， 最长频繁序列的长度maxPatternLength 和最大单机投影数据库的项数maxLocalProjDBSize。支持度阈值minSupport的定义和FPGrowth类类似，唯一差别是阈值默认值为0.1。maxPatternLength限制了最长的频繁序列的长度，越小则最后的频繁序列数越少。maxLocalProjDBSize参数是为了保护单机内存不被撑爆。如果只是是少量数据的学习，可以忽略这个参数。

　　　　从上面的描述可以看出，使用FP Tree和PrefixSpan算法没有什么门槛。学习的时候可以通过控制支持度阈值minSupport控制频繁序列的结果。而maxPatternLength可以帮忙PrefixSpan算法筛除太长的频繁序列。在分布式的大数据环境下，则需要考虑FPGrowth算法的数据分块数numPartitions，以及PrefixSpan算法的最大单机投影数据库的项数maxLocalProjDBSize。

# 3. Spark FP Tree和PrefixSpan算法使用示例

　　　　这里我们用一个具体的例子来演示如何使用Spark FP Tree和PrefixSpan算法挖掘频繁项集和频繁序列。

　　　　要使用 Spark 来学习FP Tree和PrefixSpan算法，首先需要要确保你安装好了Hadoop和Spark（版本不小于1.6），并设置好了环境变量。一般我们都是在ipython notebook\(jupyter notebook\)中学习，所以最好把基于notebook的Spark环境搭好。当然不搭notebook的Spark环境也没有关系，只是每次需要在运行前设置环境变量。

　　　　如果你没有搭notebook的Spark环境，则需要先跑下面这段代码。当然，如果你已经搭好了，则下面这段代码不用跑了。

```
import os
import sys

#下面这些目录都是你自己机器的Spark安装目录和Java安装目录
os.environ['SPARK_HOME'] = "C:/Tools/spark-1.6.1-bin-hadoop2.6/"

sys.path.append("C:/Tools/spark-1.6.1-bin-hadoop2.6/bin")
sys.path.append("C:/Tools/spark-1.6.1-bin-hadoop2.6/python")
sys.path.append("C:/Tools/spark-1.6.1-bin-hadoop2.6/python/pyspark")
sys.path.append("C:/Tools/spark-1.6.1-bin-hadoop2.6/python/lib")
sys.path.append("C:/Tools/spark-1.6.1-bin-hadoop2.6/python/lib/pyspark.zip")
sys.path.append("C:/Tools/spark-1.6.1-bin-hadoop2.6/python/lib/py4j-0.9-src.zip")
sys.path.append("C:/Program Files (x86)/Java/jdk1.8.0_102")

from pyspark import SparkContext
from pyspark import SparkConf


sc = SparkContext("local","testing")
```

　　　　在跑算法之前，建议输出Spark Context如下，如果可以正常打印内存地址，则说明Spark的运行环境搞定了。

```
print sc
```

　　　　比如我的输出是：

```
<pyspark.context.SparkContext object at 0x07D9E2B0>
```

　　　　现在我们来用数据来跑下FP Tree算法，为了和[FP Tree算法原理总结](/ml/associative/fptree.md)中的分析比照，我们使用和原理篇一样的数据项集，一样的支持度阈值20%，来训练数据。代码如下：

```
from  pyspark.mllib.fpm import FPGrowth
data = [["A", "B", "C", "E", "F","O"], ["A", "C", "G"], ["E","I"], ["A", "C","D","E","G"], ["A", "C", "E","G","L"],
       ["E","J"],["A","B","C","E","F","P"],["A","C","D"],["A","C","E","G","M"],["A","C","E","G","N"]]
rdd = sc.parallelize(data, 2)
#支持度阈值为20%
model = FPGrowth.train(rdd, 0.2, 2)
```

　　　　我们接着来看看频繁项集的结果，代码如下：

```
sorted(model.freqItemsets().collect())
```

　　　　输出即为所有 满足要求的频繁项集，大家可以和原理篇里面分析时产生的频繁项集比较。代码输出如下：

```
[FreqItemset(items=[u'A'], freq=8),
 FreqItemset(items=[u'B'], freq=2),
 FreqItemset(items=[u'B', u'A'], freq=2),
 FreqItemset(items=[u'B', u'C'], freq=2),
 FreqItemset(items=[u'B', u'C', u'A'], freq=2),
 FreqItemset(items=[u'B', u'E'], freq=2),
 FreqItemset(items=[u'B', u'E', u'A'], freq=2),
 FreqItemset(items=[u'B', u'E', u'C'], freq=2),
 FreqItemset(items=[u'B', u'E', u'C', u'A'], freq=2),
 FreqItemset(items=[u'C'], freq=8),
 FreqItemset(items=[u'C', u'A'], freq=8),
 FreqItemset(items=[u'D'], freq=2),
 FreqItemset(items=[u'D', u'A'], freq=2),
 FreqItemset(items=[u'D', u'C'], freq=2),
 FreqItemset(items=[u'D', u'C', u'A'], freq=2),
 FreqItemset(items=[u'E'], freq=8),
 FreqItemset(items=[u'E', u'A'], freq=6),
 FreqItemset(items=[u'E', u'C'], freq=6),
 FreqItemset(items=[u'E', u'C', u'A'], freq=6),
 FreqItemset(items=[u'F'], freq=2),
 FreqItemset(items=[u'F', u'A'], freq=2),
 FreqItemset(items=[u'F', u'B'], freq=2),
 FreqItemset(items=[u'F', u'B', u'A'], freq=2),
 FreqItemset(items=[u'F', u'B', u'C'], freq=2),
 FreqItemset(items=[u'F', u'B', u'C', u'A'], freq=2),
 FreqItemset(items=[u'F', u'B', u'E'], freq=2),
 FreqItemset(items=[u'F', u'B', u'E', u'A'], freq=2),
 FreqItemset(items=[u'F', u'B', u'E', u'C'], freq=2),
 FreqItemset(items=[u'F', u'B', u'E', u'C', u'A'], freq=2),
 FreqItemset(items=[u'F', u'C'], freq=2),
 FreqItemset(items=[u'F', u'C', u'A'], freq=2),
 FreqItemset(items=[u'F', u'E'], freq=2),
 FreqItemset(items=[u'F', u'E', u'A'], freq=2),
 FreqItemset(items=[u'F', u'E', u'C'], freq=2),
 FreqItemset(items=[u'F', u'E', u'C', u'A'], freq=2),
 FreqItemset(items=[u'G'], freq=5),
 FreqItemset(items=[u'G', u'A'], freq=5),
 FreqItemset(items=[u'G', u'C'], freq=5),
 FreqItemset(items=[u'G', u'C', u'A'], freq=5),
 FreqItemset(items=[u'G', u'E'], freq=4),
 FreqItemset(items=[u'G', u'E', u'A'], freq=4),
 FreqItemset(items=[u'G', u'E', u'C'], freq=4),
 FreqItemset(items=[u'G', u'E', u'C', u'A'], freq=4)]
```

　　　　接着我们来看看使用PrefixSpan类来挖掘频繁序列。为了和[PrefixSpan算法原理总结](/ml/associative/prefixspan.md)中的分析比照，我们使用和原理篇一样的数据项集，一样的支持度阈值50%，同时将最长频繁序列程度设置为4，来训练数据。代码如下：

```
from  pyspark.mllib.fpm import PrefixSpan
data = [
   [['a'],["a", "b", "c"], ["a","c"],["d"],["c", "f"]],
   [["a","d"], ["c"],["b", "c"], ["a", "e"]],
   [["e", "f"], ["a", "b"], ["d","f"],["c"],["b"]],
   [["e"], ["g"],["a", "f"],["c"],["b"],["c"]]
   ]
rdd = sc.parallelize(data, 2)
model = PrefixSpan.train(rdd, 0.5,4)
```

```
sorted(model.freqSequences().collect())
```

　　　输出即为所有满足要求的频繁序列，大家可以和原理篇里面分析时产生的频繁序列比较。代码输出如下：　

```
[FreqSequence(sequence=[[u'a']], freq=4),
 FreqSequence(sequence=[[u'a'], [u'a']], freq=2),
 FreqSequence(sequence=[[u'a'], [u'b']], freq=4),
 FreqSequence(sequence=[[u'a'], [u'b'], [u'a']], freq=2),
 FreqSequence(sequence=[[u'a'], [u'b'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'a'], [u'b', u'c']], freq=2),
 FreqSequence(sequence=[[u'a'], [u'b', u'c'], [u'a']], freq=2),
 FreqSequence(sequence=[[u'a'], [u'c']], freq=4),
 FreqSequence(sequence=[[u'a'], [u'c'], [u'a']], freq=2),
 FreqSequence(sequence=[[u'a'], [u'c'], [u'b']], freq=3),
 FreqSequence(sequence=[[u'a'], [u'c'], [u'c']], freq=3),
 FreqSequence(sequence=[[u'a'], [u'd']], freq=2),
 FreqSequence(sequence=[[u'a'], [u'd'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'a'], [u'f']], freq=2),
 FreqSequence(sequence=[[u'b']], freq=4),
 FreqSequence(sequence=[[u'b'], [u'a']], freq=2),
 FreqSequence(sequence=[[u'b'], [u'c']], freq=3),
 FreqSequence(sequence=[[u'b'], [u'd']], freq=2),
 FreqSequence(sequence=[[u'b'], [u'd'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'b'], [u'f']], freq=2),
 FreqSequence(sequence=[[u'b', u'a']], freq=2),
 FreqSequence(sequence=[[u'b', u'a'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'b', u'a'], [u'd']], freq=2),
 FreqSequence(sequence=[[u'b', u'a'], [u'd'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'b', u'a'], [u'f']], freq=2),
 FreqSequence(sequence=[[u'b', u'c']], freq=2),
 FreqSequence(sequence=[[u'b', u'c'], [u'a']], freq=2),
 FreqSequence(sequence=[[u'c']], freq=4),
 FreqSequence(sequence=[[u'c'], [u'a']], freq=2),
 FreqSequence(sequence=[[u'c'], [u'b']], freq=3),
 FreqSequence(sequence=[[u'c'], [u'c']], freq=3),
 FreqSequence(sequence=[[u'd']], freq=3),
 FreqSequence(sequence=[[u'd'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'd'], [u'c']], freq=3),
 FreqSequence(sequence=[[u'd'], [u'c'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'e']], freq=3),
 FreqSequence(sequence=[[u'e'], [u'a']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'a'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'a'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'a'], [u'c'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'b'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'c'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'f']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'f'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'f'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'e'], [u'f'], [u'c'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'f']], freq=3),
 FreqSequence(sequence=[[u'f'], [u'b']], freq=2),
 FreqSequence(sequence=[[u'f'], [u'b'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'f'], [u'c']], freq=2),
 FreqSequence(sequence=[[u'f'], [u'c'], [u'b']], freq=2)]
```

　　在训练出模型后，我们也可以调用save方法将模型存到磁盘，然后在需要的时候通过FPGrowthModel或PrefixSpanModel将模型读出来。

