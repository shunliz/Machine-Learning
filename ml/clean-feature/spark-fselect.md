基于SparkML的特征选择（Feature Selectors）三个算法（VectorSlicer、RFormula以及ChiSqSelector）结合Demo进行一下理解

# VectorSlicer算法介绍：

VectorSlicer是一个转换器输入特征向量，输出原始特征向量子集。VectorSlicer接收带有特定索引的向量列，通过对这些索引的值进行筛选得到新的向量集。可接受如下两种索引：

> 1、整数索引---代表向量中特征的的索引，setIndices\(\)
>
> 2、字符串索引---代表向量中特征的名字，这要求向量列有AttributeGroup，因为这根据Attribute来匹配名字字段

指定整数或者字符串类型都是可以的。另外，同时使用整数索引和字符串名字也是可以的。同时注意，至少选择一个特征，不能重复选择同一特征（整数索引和名字索引对应的特征不能叠）。注意如果使用名字特征，当遇到空值的时候将会报错。

输出向量将会首先按照所选的数字索引排序（按输入顺序），其次按名字排序（按输入顺序）。

**示例：**输入一个包含列名为userFeatures的DataFrame：

```
 userFeatures
------------------
 [0.0, 10.0, 0.5]
```

userFeatures是一个向量列包含3个用户特征。假设userFeatures的第一列全为0，我们希望删除它并且只选择后两项。我们可以通过索引setIndices\(1,2\)来选择后两项并产生一个新的features列：

```
 userFeatures     | features
------------------|-----------------------------
 [0.0, 10.0, 0.5] | [10.0, 0.5]
```

假设我们还有如同\["f1","f2", "f3"\]的属性，那可以通过名字setNames\("f2","f3"\)的形式来选择：

```
 userFeatures     | features
------------------|-----------------------------
 [0.0, 10.0, 0.5] | [10.0, 0.5]
 ["f1", "f2", "f3"] | ["f2", "f3"]
```

```java
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.attribute.Attribute;
import org.apache.spark.ml.attribute.AttributeGroup;
import org.apache.spark.ml.attribute.NumericAttribute;
import org.apache.spark.ml.feature.VectorSlicer;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

public class VectorSlicerDemo {
    public static void main(String[] args){
        SparkConf conf = new SparkConf().setAppName("VectorSlicer").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(sc);

        Attribute[] attributes = new Attribute[]{
                NumericAttribute.defaultAttr().withName("f1"),
                NumericAttribute.defaultAttr().withName("f2"),
                NumericAttribute.defaultAttr().withName("f3")
        };
        AttributeGroup group = new AttributeGroup("userFeatures", attributes);

        List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.sparse(3, new int[]{0, 1}, new double[]{-2.0, 2.3})),
                RowFactory.create(Vectors.dense(-2.0, 2.3, 0.0))
        );

        Dataset<Row> dataset = sqlContext
                .createDataFrame(data, (new StructType())
                .add(group.toStructField()));

        VectorSlicer vectorSlicer = new VectorSlicer()
                .setInputCol("userFeatures")
                .setOutputCol("features");

        vectorSlicer.setIndices(new int[]{1}).setNames(new String[]{"f3"});
        // or slicer.setIndices(new int[]{1, 2}), or slicer.setNames(new String[]{"f2", "f3"})

        Dataset<Row> output = vectorSlicer.transform(dataset);
        output.show(false);

//        +--------------------+-------------+
//        |userFeatures        |features     |
//        +--------------------+-------------+
//        |(3,[0,1],[-2.0,2.3])|(2,[0],[2.3])|
//        |[-2.0,2.3,0.0]      |[2.3,0.0]    |
//        +--------------------+-------------+

        sc.stop();
    }
}
```

# **RFormula算法介绍：**

RFormula通过R模型公式来选择列。支持R操作中的部分操作，包括‘~’, ‘.’, ‘:’, ‘+’以及‘-‘，基本操作如下：

> 1、 ~分隔目标和对象
>
> 2、 +合并对象，“+0”意味着删除空格
>
> 3、-删除一个对象，“-1”表示删除空格
>
> 4、 :交互（数值相乘，类别二值化）
>
> 5、 . 除了目标列的全部列

假设a和b为两列：

> 1、 y ~ a + b表示模型y ~ w0 + w1 \* a +w2 \* b其中w0为截距，w1和w2为相关系数
>
> 2、 y ~a + b + a:b – 1表示模型y ~ w1\* a + w2 \* b + w3 \* a \* b，其中w1，w2，w3是相关系数

RFormula产生一个向量特征列以及一个double或者字符串标签列。如果用R进行线性回归，则对String类型的输入列进行one-hot编码、对数值型的输入列进行double类型转化。如果类别列是字符串类型，它将通过StringIndexer转换为double类型。如果标签列不存在，则输出中将通过规定的响应变量创造一个标签列。

**示例：**假设我们有一个DataFrame含有id,country, hour和clicked四列：

```
id | country | hour | clicked
---|---------|------|---------
 7 | "US"    | 18   | 1.0
 8 | "CA"    | 12   | 0.0
 9 | "NZ"    | 15   | 0.0
```

如果我们使用RFormula公式clicked ~ country+ hour，则表明我们希望基于country和hour预测clicked，通过转换我们可以得到如DataFrme：

```
id | country | hour | clicked | features         | label
---|---------|------|---------|------------------|-------
 7 | "US"    | 18   | 1.0     | [0.0, 0.0, 18.0] | 1.0
 8 | "CA"    | 12   | 0.0     | [0.0, 1.0, 12.0] | 0.0
 9 | "NZ"    | 15   | 0.0     | [1.0, 0.0, 15.0] | 0.0
```

```java
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.RFormula;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.types.DataTypes.*;

public class RFormulaDemo {
    public static void main(String[] args){
        SparkConf conf = new SparkConf().setAppName("RFoumula").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(sc);

        List<Row> data = Arrays.asList(
                RowFactory.create(7,"US",18,1.0),
                RowFactory.create(8,"CA",12,0.0),
                RowFactory.create(9,"NZ",15,0.0)
        );

        StructType schema = createStructType(new StructField[]{
                createStructField("id", IntegerType, false),
                createStructField("country", StringType, false),
                createStructField("hour", IntegerType, false),
                createStructField("clicked", DoubleType, false)
        });

        Dataset<Row> dataset = sqlContext.createDataFrame(data,schema);
        RFormula formula = new RFormula()
                .setFormula("clicked ~ country + hour")
                .setFeaturesCol("features")
                .setLabelCol("label");
        Dataset<Row> output = formula.fit(dataset).transform(dataset);
        output.select("features", "label").show(false);

//        +--------------+-----+
//        |features      |label|
//        +--------------+-----+
//        |[0.0,0.0,18.0]|1.0  |
//        |[1.0,0.0,12.0]|0.0  |
//        |[0.0,1.0,15.0]|0.0  |
//        +--------------+-----+

        sc.stop();
    }
}
```

# ChiSqSelector算法介绍：

ChiSqSelector代表卡方特征选择。它适用于带有类别特征的标签数据。ChiSqSelector根据独立卡方检验，然后选取类别标签主要依赖的特征。它类似于选取最有预测能力的特征。它支持三种特征选取方法：

> 1、numTopFeatures：通过卡方检验选取最具有预测能力的Top\(num\)个特征；
>
> 2、percentile：类似于上一种方法，但是选取一小部分特征而不是固定\(num\)个特征；
>
> 3、fpr:选择P值低于门限值的特征，这样就可以控制false positive rate来进行特征选择；

默认情况下特征选择方法是numTopFeatures\(50\)，可以根据setSelectorType\(\)选择特征选取方法。

**示例：**假设我们有一个DataFrame含有id,features和clicked三列，其中clicked为需要预测的目标：

```
id | features              | clicked
---|-----------------------|---------
 7 | [0.0, 0.0, 18.0, 1.0] | 1.0
 8 | [0.0, 1.0, 12.0, 0.0] | 0.0
 9 | [1.0, 0.0, 15.0, 0.1] | 0.0
```

如果我们使用ChiSqSelector并设置numTopFeatures为1，根据标签clicked，features中最后一列将会是最有用特征：

```
id | features              | clicked | selectedFeatures
---|-----------------------|---------|------------------
 7 | [0.0, 0.0, 18.0, 1.0] | 1.0     | [1.0]
 8 | [0.0, 1.0, 12.0, 0.0] | 0.0     | [0.0]
 9 | [1.0, 0.0, 15.0, 0.1] | 0.0     | [0.1]
```

```java
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.ChiSqSelector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

public class ChiSqSelectorDemo {
    public static void main(String[] args){
        SparkConf conf = new SparkConf().setAppName("Demo").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(sc);

        JavaRDD<Row> data = sc.parallelize(Arrays.asList(
                RowFactory.create(7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
                RowFactory.create(8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
                RowFactory.create(9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
        ));

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
                new StructField("clicked", DataTypes.DoubleType, false, Metadata.empty())
        });


        Dataset<Row> df = sqlContext.createDataFrame(data, schema);

        ChiSqSelector selector = new ChiSqSelector()
                .setNumTopFeatures(2)
                .setFeaturesCol("features")
                .setLabelCol("clicked")
                .setOutputCol("selectedFeatures");

        Dataset<Row> result = selector.fit(df).transform(df);
        result.show(false);

//        +---+------------------+-------+----------------+
//        |id |features          |clicked|selectedFeatures|
//        +---+------------------+-------+----------------+
//        |7  |[0.0,0.0,18.0,1.0]|1.0    |[18.0,1.0]      |
//        |8  |[0.0,1.0,12.0,0.0]|0.0    |[12.0,0.0]      |
//        |9  |[1.0,0.0,15.0,0.1]|0.0    |[15.0,0.1]      |
//        +---+------------------+-------+----------------+

        sc.stop();
    }
}
```



