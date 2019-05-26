# Pandas

---


```python
import pandas as pd
import numpy as np
```

Pandas 数据类型


```python
s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])

data = {'Country': ['Belgium', 'India', 'Brazil'],
'Capital': ['Brussels', 'New Delhi', 'Brasília'],
'Population': [11190846, 1303171035, 207847528]}
df = pd.DataFrame(data,
columns=['Country', 'Capital', 'Population'])

#Pivvot,
data = {'Date': ['2016-03-01', '2016-03-02', '2016-03-01','2016-03-03','2016-03-02',
                 '2016-03-03'],
'Type': ['a', 'b', 'c','a','a','c'],
'Value': [11.432, 13.031, 20.784,99.906,1.303,20.784]}
df2 = pd.DataFrame(data,
columns=['Date', 'Type', 'Value'])
df3= df2.pivot(index='Date',
columns='Type',
values='Value')
print df2
print df3
```

             Date Type   Value
    0  2016-03-01    a  11.432
    1  2016-03-02    b  13.031
    2  2016-03-01    c  20.784
    3  2016-03-03    a  99.906
    4  2016-03-02    a   1.303
    5  2016-03-03    c  20.784
    Type             a       b       c
    Date                              
    2016-03-01  11.432     NaN  20.784
    2016-03-02   1.303  13.031     NaN
    2016-03-03  99.906     NaN  20.784
    


```python
df4 = pd.pivot_table(df2,values='Value',index='Date',columns=['Type'])
print df2
print df4
```

             Date Type   Value
    0  2016-03-01    a  11.432
    1  2016-03-02    b  13.031
    2  2016-03-01    c  20.784
    3  2016-03-03    a  99.906
    4  2016-03-02    a   1.303
    5  2016-03-03    c  20.784
    Type             a       b       c
    Date                              
    2016-03-01  11.432     NaN  20.784
    2016-03-02   1.303  13.031     NaN
    2016-03-03  99.906     NaN  20.784
    


```python
df4 = pd.pivot_table(df2,
values='Value',
index='Date',
columns=['Type'])
print df4
```

    Type             a       b       c
    Date                              
    2016-03-01  11.432     NaN  20.784
    2016-03-02   1.303  13.031     NaN
    2016-03-03  99.906     NaN  20.784
    


```python
df5=pd.melt(df2,
id_vars=["Date"],
value_vars=["Type", "Value"],
value_name="Observations")
print df5
```

              Date variable Observations
    0   2016-03-01     Type            a
    1   2016-03-02     Type            b
    2   2016-03-01     Type            c
    3   2016-03-03     Type            a
    4   2016-03-02     Type            a
    5   2016-03-03     Type            c
    6   2016-03-01    Value       11.432
    7   2016-03-02    Value       13.031
    8   2016-03-01    Value       20.784
    9   2016-03-03    Value       99.906
    10  2016-03-02    Value        1.303
    11  2016-03-03    Value       20.784
    

数据丢弃


```python
s.drop(['a','c'])
df.drop('Country', axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>



排序


```python
df.sort_index()
df.sort_values(by='Country')
df.rank()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



获取数据信息


```python
df.shape
df.index
df.columns
df.info()
df.count()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 3 columns):
    Country       3 non-null object
    Capital       3 non-null object
    Population    3 non-null int64
    dtypes: int64(1), object(2)
    memory usage: 144.0+ bytes
    




    Country       3
    Capital       3
    Population    3
    dtype: int64



数据摘要


```python
df.sum()
df.cumsum()
#df.min()/df.max()
#df.idxmin()/df.idxmax()
df.describe()
df.mean()
df.median()
```




    Population    207847528.0
    dtype: float64



选择


```python
s['b']
df[1:]

df.iloc[[0],[0]]
df.iat([0])

#df.loc[[0], ['Country']]
#df.at([0], ['Country'])

df.ix[2]
df.ix[:,'Capital']
df.ix[1,'Capital']

#Boolean Indexing
s[~(s > 1)]
s[(s < -1) | (s > 2)]
df[df['Population']>1200000000]

s['a'] = 6

#Selecting
df3.loc[:,(df3>1).any()]
df3.loc[:,(df3>1).all()]
df3.loc[:,df3.isnull().any()]
df3.loc[:,df3.notnull().all()]
#Indexing With isin
df[(df.Country.isin(df2.Type))]
df3.filter(items=["a","b"])
df.select(lambda x: not x%5)
#Where
s.where(s > 0)
#Query
#df.query('second > first')

#Setting/Resetting Index
df.set_index('Country')
df4 = df.reset_index()
df = df.rename(index=str,columns={"Country":"cntry","Capital":"cptl",
                                  "Population":"ppltn"})
s2 = s.reindex(['a','c','d','e','b'])
```


```python
ss= df.reindex(range(4),method='ffill')
print ss
```

      cntry cptl  ppltn
    0   NaN  NaN    NaN
    1   NaN  NaN    NaN
    2   NaN  NaN    NaN
    3   NaN  NaN    NaN
    


```python
s3 = s.reindex(range(5),method='bfill')
print s3
```

    0    6
    1    6
    2    6
    3    6
    4    6
    dtype: int64
    

数据聚合


```python
#Aggregation
df2.groupby(by=['Date','Type']).mean()
df4.groupby(level=0).sum()
print df4
#df4.groupby(level=0).agg({'Capital':lambda x:sum(x)/len(x), 'Population': np.sum})
#Transformation
#customSum = lambda x: (x+x%2)
#df4.groupby(level=0).transform(customSum)
```

       index  Country    Capital  Population
    0      0  Belgium   Brussels    11190846
    1      1    India  New Delhi  1303171035
    2      2   Brazil   Brasília   207847528
    

帮助

help(pd.Series.loc)

使用函数


```python
f=lambda x:x*2
df.apply(f)
df.applymap(f)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cntry</th>
      <th>cptl</th>
      <th>ppltn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BelgiumBelgium</td>
      <td>BrusselsBrussels</td>
      <td>22381692</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IndiaIndia</td>
      <td>New DelhiNew Delhi</td>
      <td>2606342070</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BrazilBrazil</td>
      <td>BrasíliaBrasília</td>
      <td>415695056</td>
    </tr>
  </tbody>
</table>
</div>



数据对齐


```python
s3 = pd.Series([7, -2, 3], index=['a', 'c', 'd'])
s + s3
```




    a    13.0
    b     NaN
    c     5.0
    d     7.0
    dtype: float64



输入输出

#读取和写入csv
pd.read_csv('file.csv', header=None, nrows=5)
df.to_csv('myDataFrame.csv')

#读取和写入excel
pd.read_excel('file.xlsx')
pd.to_excel('dir/myDataFrame.xlsx', sheet_name='Sheet1')
#从多个表单读取数据
xlsx = pd.ExcelFile('file.xls')
df = pd.read_excel(xlsx, 'Sheet1')

#从SQL查询或者数据库表读取和写入数据
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
pd.read_sql("SELECT * FROM my_table;", engine)
pd.read_sql_table('my_table', engine)
pd.read_sql_query("SELECT * FROM my_table;", engine)

数据合并


```python
dict1 = {'X1': ['a', 'b', 'c'],
'X2': ['11.432', '1.303', '99.906']}

dict2 = {'X1': ['a', 'b', 'd'],
'X3': ['20.784', 'NaN', '20.784']}

```


```python
data1 = pd.DataFrame(dict1,
columns=['X1', 'X2'])
data2 = pd.DataFrame(dict2,
columns=['X1', 'X3'])

pd.merge(data1,
data2,
how='left',
on='X1')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>99.906</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(data1,
data2,
how='right',
on='X1')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(data1,
data2,
how='inner',
on='X1')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(data1,
data2,
how='outer',
on='X1')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>99.906</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>



join


```python
#help(df.join)

caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                        'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                      'B': ['B0', 'B1', 'B2']})
caller.join(other, lsuffix='_caller', rsuffix='_other')
print caller
#data1.set_index('X1')
#data2.set_index('X1')
#data1.join(data2, lsuffix='data1', rsuffix='data2', how='right')

```

        A key
    0  A0  K0
    1  A1  K1
    2  A2  K2
    3  A3  K3
    4  A4  K4
    5  A5  K5
    


```python
caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                        'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                      'B': ['B0', 'B1', 'B2']})
caller.set_index('key').join(other.set_index('key'))
print caller
```

        A key
    0  A0  K0
    1  A1  K1
    2  A2  K2
    3  A3  K3
    4  A4  K4
    5  A5  K5
    


```python
caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                        'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                      'B': ['B0', 'B1', 'B2']})
caller.join(other.set_index('key'), on='key')
print caller
```

        A key
    0  A0  K0
    1  A1  K1
    2  A2  K2
    3  A3  K3
    4  A4  K4
    5  A5  K5
    

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



