

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer} 
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.MinMaxScaler 
import org.apache.spark.ml.linalg.Vectors 
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._ 
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder,TrainValidationSplit}
import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
// $example off$
import org.apache.spark.sql.SparkSession

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
//import sqlContext.implicits._
//import sqlContext._


object Auto_MPG {

  def main(args: Array[String]): Unit = {

    //disable logging
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    
    
    //1. Data Exploration:
 
    /**************************************************
    
[edureka_918210@ip-20-0-41-164 Auto_MPG]$ hadoop fs -ls /user/edureka_918210/Auto_MPG
Found 1 items
-rw-r--r--   3 edureka_918210 hadoop      18045 2021-01-10 10:07 /user/edureka_918210/Auto_MPG/auto-mpg.csv
[edureka_918210@ip-20-0-41-164 Auto_MPG]$ 
     */
    
val spark = SparkSession.builder().appName("Auto_MPG-spark").master("local[*]").getOrCreate();

    
//image_id,label 
    
val autompg = spark.read.option("delimiter",",").option("inferSchema", true).csv("/user/edureka_918210/Auto_MPG/auto-mpg.csv").toDF("mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name") 

autompg.show(100)
autompg.printSchema()

val autompg_uniqueid = autompg.withColumn("UniqueID",monotonicallyIncreasingId)
autompg_uniqueid.printSchema()



/*******************************************************************************************

scala> autompg_uniqueid.printSchema()
root
 |-- mpg: double (nullable = true)
 |-- cylinders: integer (nullable = true)
 |-- displacement: double (nullable = true)
 |-- horsepower: string (nullable = true)
 |-- weight: integer (nullable = true)
 |-- acceleration: double (nullable = true)
 |-- model_year: integer (nullable = true)
 |-- origin: integer (nullable = true)
 |-- car_name: string (nullable = true)
 |-- UniqueID: long (nullable = false)

 ****************************************************************************************************/

//Compute basic statistics for numeric columns - count, mean, standard deviation, min, and max

/********************************************
scala> autompg_uniqueid.count()
res11: Long = 398


*********************************/

//2. Data Cleaning: 

val autompg_filter = autompg_uniqueid.selectExpr("mpg", "cast(cylinders as double) cylinders", "cast(displacement as double) displacement", "cast(weight as double) weight", "cast(acceleration as double) acceleration", "cast(model_year as double) model_year","cast(UniqueID as double) UniqueID")

/**************************************************
 * 
 root
 |-- mpg: double (nullable = true)
 |-- cylinders: double (nullable = true)
 |-- displacement: double (nullable = true)
 |-- weight: double (nullable = true)
 |-- acceleration: double (nullable = true)
 |-- model_year: double (nullable = true)
 |-- UniqueID: double (nullable = false)
 
 */

val DFAssembler = new VectorAssembler().setInputCols(Array("mpg","cylinders","displacement","weight","acceleration","model_year","UniqueID")).setOutputCol("features")

// Elimination null values

val features = DFAssembler.transform(autompg_filter.na.drop)
features.show(2)

val labeledfeatures = new StringIndexer().setInputCol("UniqueID").setOutputCol("label")
val df3 = labeledfeatures.fit(features).transform(features)



//3. Model Building:

val kmeans = new org.apache.spark.ml.clustering.KMeans().setK(4).setFeaturesCol("features").setPredictionCol("prediction")


val model = kmeans.fit(df3)

model.clusterCenters.foreach(println)
val categories = model.transform(df3)
categories.createOrReplaceTempView("c")

/****************************************************************

scala> spark.sql("select * from c").show(false)
+----+---------+------------+------+------------+----------+--------+--------------------------------------+-----+----------+
|mpg |cylinders|displacement|weight|acceleration|model_year|UniqueID|features                              |label|prediction|
+----+---------+------------+------+------------+----------+--------+--------------------------------------+-----+----------+
|18.0|8.0      |307.0       |3504.0|12.0        |70.0      |0.0     |[18.0,8.0,307.0,3504.0,12.0,70.0,0.0] |112.0|3         |
|15.0|8.0      |350.0       |3693.0|11.5        |70.0      |1.0     |[15.0,8.0,350.0,3693.0,11.5,70.0,1.0] |271.0|3         |
|18.0|8.0      |318.0       |3436.0|11.0        |70.0      |2.0     |[18.0,8.0,318.0,3436.0,11.0,70.0,2.0] |185.0|3         |
|16.0|8.0      |304.0       |3433.0|12.0        |70.0      |3.0     |[16.0,8.0,304.0,3433.0,12.0,70.0,3.0] |122.0|3         |
|17.0|8.0      |302.0       |3449.0|10.5        |70.0      |4.0     |[17.0,8.0,302.0,3449.0,10.5,70.0,4.0] |215.0|3         |
|15.0|8.0      |429.0       |4341.0|10.0        |70.0      |5.0     |[15.0,8.0,429.0,4341.0,10.0,70.0,5.0] |370.0|0         |
|14.0|8.0      |454.0       |4354.0|9.0         |70.0      |6.0     |[14.0,8.0,454.0,4354.0,9.0,70.0,6.0]  |30.0 |0         |
|14.0|8.0      |440.0       |4312.0|8.5         |70.0      |7.0     |[14.0,8.0,440.0,4312.0,8.5,70.0,7.0]  |368.0|0         |
|14.0|8.0      |455.0       |4425.0|10.0        |70.0      |8.0     |[14.0,8.0,455.0,4425.0,10.0,70.0,8.0] |148.0|0         |
|15.0|8.0      |390.0       |3850.0|8.5         |70.0      |9.0     |[15.0,8.0,390.0,3850.0,8.5,70.0,9.0]  |292.0|3         |
|15.0|8.0      |383.0       |3563.0|10.0        |70.0      |10.0    |[15.0,8.0,383.0,3563.0,10.0,70.0,10.0]|181.0|3         |
|14.0|8.0      |340.0       |3609.0|8.0         |70.0      |11.0    |[14.0,8.0,340.0,3609.0,8.0,70.0,11.0] |213.0|3         |
|15.0|8.0      |400.0       |3761.0|9.5         |70.0      |12.0    |[15.0,8.0,400.0,3761.0,9.5,70.0,12.0] |51.0 |3         |
|14.0|8.0      |455.0       |3086.0|10.0        |70.0      |13.0    |[14.0,8.0,455.0,3086.0,10.0,70.0,13.0]|239.0|3         |
|24.0|4.0      |113.0       |2372.0|15.0        |70.0      |14.0    |[24.0,4.0,113.0,2372.0,15.0,70.0,14.0]|360.0|1         |
|22.0|6.0      |198.0       |2833.0|15.5        |70.0      |15.0    |[22.0,6.0,198.0,2833.0,15.5,70.0,15.0]|54.0 |2         |
|18.0|6.0      |199.0       |2774.0|15.5        |70.0      |16.0    |[18.0,6.0,199.0,2774.0,15.5,70.0,16.0]|196.0|2         |
|21.0|6.0      |200.0       |2587.0|16.0        |70.0      |17.0    |[21.0,6.0,200.0,2587.0,16.0,70.0,17.0]|274.0|2         |
|27.0|4.0      |97.0        |2130.0|14.5        |70.0      |18.0    |[27.0,4.0,97.0,2130.0,14.5,70.0,18.0] |361.0|1         |
|26.0|4.0      |97.0        |1835.0|20.5        |70.0      |19.0    |[26.0,4.0,97.0,1835.0,20.5,70.0,19.0] |116.0|1         |
+----+---------+------------+------+------------+----------+--------+--------------------------------------+-----+----------+
only showing top 20 rows

**********************************************************/

/******************
 * RESULT JAR
 * [edureka_918210@ip-20-0-41-164 Auto_MPG]$ hadoop fs -ls /user/edureka_918210/Auto_MPG
Found 3 items
-rw-r--r--   3 edureka_918210 hadoop      16667 2021-01-10 10:41 /user/edureka_918210/Auto_MPG/Auto_MPG-1.0-SNAPSHOT.jar
-rw-r--r--   3 edureka_918210 hadoop      18045 2021-01-10 10:07 /user/edureka_918210/Auto_MPG/auto-mpg.csv
-rw-r--r--   3 edureka_918210 hadoop       3098 2021-01-10 10:41 /user/edureka_918210/Auto_MPG/predictions.txt
[edureka_918210@ip-20-0-41-164 Auto_MPG]$ 
[edureka_918210@ip-20-0-41-164 Auto_MPG]$ 
 */


  }
  
}


    
    
