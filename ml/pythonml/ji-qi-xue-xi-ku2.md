# Pandas

---

# 

# pySpark

---

Initializing Spark


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


```python
Loading Data
```


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


```python
Selecting Data
```


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

''''''
('a', 7)
('b', 2)
('a', 2)
'''''''


```python

```

# 

# 



