# Pandas

---

# 

# pySpark

---


初始化spark


```python
#SparkContext
from pyspark import SparkContext
sc = SparkContext(master = 'local[2]')
```


```python
#Calculations With Variables
sc.version
sc.pythonVer
sc.master
str(sc.sparkHome)
str(sc.sparkUser())
sc.appName
sc.applicationId
sc.defaultParallelism
sc.defaultMinPartitions
```


```python
#Configuration
from pyspark import SparkConf, SparkContext
conf = (SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
```

加载数据


```python
#Parallelized Collections
rdd = sc.parallelize([('a',7),('a',2),('b',2)])
rdd2 = sc.parallelize([('a',2),('d',1),('b',1)])
rdd3 = sc.parallelize(range(100))
rdd4 = sc.parallelize([("a",["x","y","z"]),("b",["p", "r"])])
```


```python
#External Data
textFile = sc.textFile("/my/directory/*.txt")
textFile2 = sc.wholeTextFiles("/my/directory/")
```

选择数据


```python
#Getting
rdd.collect()  #[('a', 7), ('a', 2), ('b', 2)]
rdd.take(2)    #[('a', 7), ('a', 2)]
rdd.first()    #('a', 7)
rdd.top(2)     #[('b', 2), ('a', 7)]

#Sampling
rdd3.sample(False, 0.15, 81).collect() #[3,4,27,31,40,41,42,43,60,76,79,80,86,97]

#Filtering
rdd.filter(lambda x: "a" in x).collect() #[('a',7),('a',2)]
rdd5.distinct().collect()  #['a',2,'b',7]
rdd.keys().collect() #['a', 'a', 'b']

#Iterating
def g(x): print(x)
rdd.foreach(g)
```

('a', 7)
('b', 2)
('a', 2)

获取RDD信息：基本信息


```python
rdd.getNumPartitions()
rdd.count()
rdd.countByKey()
rdd.countByValue()
rdd.collectAsMap()
rdd3.sum()
sc.parallelize([]).isEmpty()
```

获取RDD信息：概要信息


```python
rdd3.max()
rdd3.min()
rdd3.mean()
rdd3.stdev()
rdd3.variance()
rdd3.histogram(3)
```

使用函数


```python
rdd.map(lambda x:x+(x[1],x[0]))
rdd5=rdd.flatMap(lambda x:x+(x[1],x[0]))
rdd5.collect()
rdd4.flatMapValues(lambda x:x).collect()
```

数学操作


```python
rdd.subtracrt(rdd2)  #返回差集
rdd2.subtractByKey(rdd)  #返回key的差集
rdd.cartesian(rdd2).collect() 
```

排序


```python
rdd2.sortBy(lambda x:x[1]).collect()
rdd2.sortByKey()
```

数据变形


```python
rdd.repartition(4)
rddd.coalesce(1)
```

保存


```python
rdd.saveAsTextFile("rdd.txt")
rdd.saveAsHadoopFile("hdfs://xxxxx.",'org.apache.hadoop.mapred.TextOutputFormat')
```

停止


```python
sc.stop()
```

执行


```python
$ ./bin/spark-submit examples/src/main/python/pi.py
```


# 



