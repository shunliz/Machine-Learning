## Data Preparation

You must pre-process your raw data before you model your problem. The specific preparation may depend on the data that you have available and the machine learning algorithms you want to use.

Sometimes, pre-processing of data can lead to unexpected improvements in model accuracy. This may be because a relationship in the data has been simplified or unobscured.

[Data preparation](http://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/)is an important step and you should experiment with data pre-processing steps that are appropriate for your data to see if you can get that desirable boost in model accuracy.

There are three types of pre-processing you can consider for your data:

* Add attributes to your data
* Remove attributes from your data
* Transform attributes in your data

We will dive into each of these three types of pre-process and review some specific examples of operations that you can perform.

## Add Data Attributes

Advanced models can extract the relationships from complex attributes, although some models require those relationships to be spelled out plainly. Deriving new attributes from your training data to include in the modeling process can give you a boost in model performance.

* **Dummy Attributes**
  : Categorical attributes can be converted into n-binary attributes, where n is the number of categories \(or levels\) that the attribute has. These denormalized or decomposed attributes are known as dummy attributes or dummy variables.
* **Transformed Attribute**
  : A transformed variation of an attribute can be added to the dataset in order to allow a linear method to exploit possible linear and non-linear relationships between attributes. Simple transforms like log, square and square root can be used.
* **Missing Data**
  : Attributes with missing data can have that missing data imputed using a reliable method, such as k-nearest neighbors.

## Remove Data Attributes

Some methods perform poorly with redundant or duplicate attributes. You can get a boost in model accuracy by removing attributes from your data.

* **Projection**
  : Training data can be projected into lower dimensional spaces, but still characterize the inherent relationships in the data. A popular approach is Principal Component Analysis \(PCA\) where the principal components found by the method can be taken as a reduced set of input attributes.
* **Spatial Sign**
  : A spatial sign projection of the data will transform data onto the surface of a multidimensional sphere. The results can be used to highlight the existence of outliers that can be modified or removed from the data.
* **Correlated Attributes**
  : Some algorithms degrade in importance with the existence of highly correlated attributes. Pairwise attributes with high correlation can be identified and the most correlated attributes can be removed from the data.

## Transform Data Attributes

Transformations of training data can reduce the skewness of data as well as the prominence of outliers in the data. Many models expect data to be transformed before you can apply the algorithm.

* **Centering**
  : Transform the data so that it has a mean of zero and a standard deviation of one. This is typically called data standardization.
* **Scaling**
  : A standard scaling transformation is to map the data from the original scale to a scale between zero and one. This is typically called data normalization.
* **Remove Skew**
  : Skewed data is data that has a distribution that is pushed to one side or the other \(larger or smaller values\) rather than being normally distributed. Some methods assume normally distributed data and can perform better if the skew is removed. Try replacing the attribute with the log, square root or inverse of the values.
* **Box-Cox**
  : A Box-Cox transform or family of transforms can be used to reliably adjust data to remove skew.
* **Binning**
  : Numeric data can be made discrete by grouping values into bins. This is typically called data discretization. This process can be performed manually, although is more reliable if performed systematically and automatically using a heuristic that makes sense in the domain.

## Summary

Data pre-process is an important step that can be required to prepare raw data for modeling, to meet the expectations of data for a specific machine learning algorithms, and can give unexpected boosts in model accuracy.

In this post we discovered three groups of data pre-processing methods:

* Adding Attributes
* Removing Attributes
* Transforming Attributes



