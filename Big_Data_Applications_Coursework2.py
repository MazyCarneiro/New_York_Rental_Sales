################################################################
# BIG DATA APPLICATIONS COURSEWORK 2                            #
# DUE DATE - 18th April 2019                                    #
# LECTURER - AARON GEROW                                        #
#################################################################

# PRINT START TIME
import datetime
datetime.datetime.now()

timestart = datetime.datetime.now()

#Libraries used for the assignment
from pyspark  import SparkContext, SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes
from pyspark.mllib.classification import NaiveBayesModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
import pandas as pd

# Create my_spark
my_spark = SparkSession.builder.appName('Big Data Applications - Coursework2').getOrCreate()
# Print my_spark
print(my_spark)

path = 'NewYork_RentalSales_MAIN.csv'
df = my_spark.read.csv(path,header=True,inferSchema=True)

df = df.drop('sale_price')

df.printSchema()

df.head(5)

len(df.columns), df.columns

pd.DataFrame(df.take(5), columns=df.columns).transpose()

df.describe().toPandas()
df.groupby('Above_Below_median').count().toPandas()

df.rdd.getNumPartitions()

# Numerical columns that had missing values could have been filled using K nearest neighbor
# and so could the categorical columns too but in the case of this dataset, a rental appartment
# not having a commercial unit or not having a police precinct does make sense. These NULL values
# in these columns do have significance hence the author has decided not to fill in the
# missing values.

# Let's see how many categorical features we have:
cat_cols = [item[0] for item in df.dtypes if item[1].startswith('string')]
print(str(len(cat_cols)) + '  categorical features')

# Let's see how many numerical features we have:
num_cols = [item[0] for item in df.dtypes if item[1].startswith('int') | item[1].startswith('double')][1:]
print(str(len(num_cols)) + '  numerical features')

Data = df.rdd.map(lambda x:(Vectors.dense(x[0:-1]), x[-1])).toDF(["features", "label"])
Data.show()
pd.DataFrame(Data.take(5), columns=Data.columns)

testset,trainset = Data.randomSplit([0.3,0.7], seed=25)
print("Training Dataset Count: " + str(trainset.count()))
print("Test Dataset Count: " + str(testset.count()))

### GENERALIZED LINEAR REGRESSION FOR FEATURE SELECTION
from pyspark.ml.regression import GeneralizedLinearRegression
glr = GeneralizedLinearRegression(predictionCol="Predicted_median", labelCol="label", featuresCol="features",family="binomial", link="logit", maxIter=10,regParam=0.01)
model = glr.fit(Data)
summary = model.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("P Values: " + str(summary.pValues))

#Removing all the columns that had a p-value above 0.05
vs = VectorSlicer(inputCol="features", outputCol="selected_features", indices=[0,2,9,18,21,23,24,26,27,28,31,32,37,41])
Training_set= vs.transform(trainset)
Test_set = vs.transform(testset)

#### LOGISTIC REGRESSION
logReg = LogisticRegression(predictionCol="Predicted_median", labelCol="label", featuresCol="features", maxIter=20,regParam=0.01, elasticNetParam=0.8, family="binomial")
logReg_model = logReg.fit(Training_set)
trainingSummary = logReg_model.summary
roc = trainingSummary.roc.toPandas()
print('Training set ROC: ' + str(trainingSummary.areaUnderROC))
predictions = logReg_model.transform(Test_set)
predictions.select('features', 'label', 'rawPrediction', 'Predicted_median', 'probability').show(10)
evaluator = BinaryClassificationEvaluator()
print("Test_SET (Area Under ROC): " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))


### GRADIENT BOOSTING
from pyspark.ml.classification import GBTClassifier

gb = GBTClassifier(predictionCol="Predicted_median", labelCol="label", featuresCol="features", maxIter=20)
gbModel = gb.fit(Training_set)
gb_predictions = gbModel.transform(Test_set)
gb_predictions.select('features', 'label', 'rawPrediction', 'Predicted_median', 'probability').show(10)
evaluator = BinaryClassificationEvaluator()
print("Test_SET (Area Under ROC): " + str(evaluator.evaluate(gb_predictions, {evaluator.metricName: "areaUnderROC"})))

### RANDOM FORESTS

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(predictionCol="Predicted_median",labelCol="label",featuresCol="features",numTrees = 40, maxDepth = 30)
rfModel = rf.fit(Training_set)
rf_predictions = rfModel.transform(Test_set)
rf_predictions.filter(rf_predictions['Predicted_median'] == 0) \
           .select("features","label","Predicted_median","probability") \
           .orderBy("probability", ascending=False) \
           .show(n = 10, truncate = 30)
evaluator = BinaryClassificationEvaluator()
print("Test_SET (Area Under ROC): " + str(evaluator.evaluate(rf_predictions, {evaluator.metricName: "areaUnderROC"})))

## DECISION TREES
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(predictionCol='Predicted_median',featuresCol = 'features', labelCol = 'label',maxDepth = 30)
dtModel = dt.fit(Training_set)
dt_predictions = dtModel.transform(Test_set)
dt_predictions.select('features', 'label', 'rawPrediction', 'Predicted_median', 'probability').show(10)
evaluator = BinaryClassificationEvaluator()
print("Test_SET (Area Under ROC): " + str(evaluator.evaluate(dt_predictions, {evaluator.metricName: "areaUnderROC"})))

# LETS TRY CROSS VALIDATION ON THE GRADIENT BOOSTING MODEL TO SEE IF THE PERFORMANCE IMPROVES.

#GRADIENT BOOSTING WITH CROSS VALIDATION
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder().addGrid(gb.maxDepth, [2, 4, 10]).addGrid(gb.maxBins, [10, 20]).addGrid(gb.maxIter, [10, 25]).build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
# Run cross validations.
cvModel = cv.fit(Training_set)
gb_cv_predictions = cvModel.transform(Test_set)
gb_cv_predictions.select('features', 'label', 'rawPrediction', 'Predicted_median', 'probability').show(10)
evaluator = BinaryClassificationEvaluator()
print("Test_SET (Area Under ROC): " + str(evaluator.evaluate(gb_cv_predictions, {evaluator.metricName: "areaUnderROC"})))

