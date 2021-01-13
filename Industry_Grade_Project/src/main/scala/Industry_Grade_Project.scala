

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

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.{StructType, StructField, StringType};


object Industry_Grade_project {

  def main(args: Array[String]): Unit = {

    //disable logging
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    
val spark = SparkSession.builder().appName("Industry_Grade_Project-spark").master("local[*]").getOrCreate();

spark.read.textFile("/user/edureka_918210/project_retailcart/batchdata/retailcart_calendar_details.txt").createOrReplaceTempView("retailcart_calendar_details");

val retailcart_calendar = 
spark.sql(""" Select 
split(value,'\t')[0] as calendar_date,
split(value,'\t')[1] as date_desc,
split(value,'\t')[2] as week_day_nbr,
split(value,'\t')[3] as week_number,
split(value,'\t')[4] as week_name,
split(value,'\t')[5] as year_week_number,
split(value,'\t')[6] as month_number,
split(value,'\t')[7] as month_name,
split(value,'\t')[8] as quarter_number,
split(value,'\t')[9] as quarter_name,
split(value,'\t')[10] as half_year_number,
split(value,'\t')[11] as half_year_name,
split(value,'\t')[12] as geo_region_cd
from retailcart_calendar_details """)

/***********************************************
 * 
scala> retailcart_calendar.printSchema
root
 |-- calendar_date: string (nullable = true)
 |-- date_desc: string (nullable = true)
 |-- week_day_nbr: string (nullable = true)
 |-- week_number: string (nullable = true)
 |-- week_name: string (nullable = true)
 |-- year_week_number: string (nullable = true)
 |-- month_number: string (nullable = true)
 |-- month_name: string (nullable = true)
 |-- quarter_number: string (nullable = true)
 |-- quarter_name: string (nullable = true)
 |-- half_year_number: string (nullable = true)
 |-- half_year_name: string (nullable = true)
 |-- geo_region_cd: string (nullable = true)
 
 */
/*****************************************
scala> retailcart_calendar.show(3)
+-------------+--------------------+------------+-----------+---------+----------------+------------+----------+--------------+------------+----------------+------
--------+-------------+
|calendar_date|           date_desc|week_day_nbr|week_number|week_name|year_week_number|month_number|month_name|quarter_number|quarter_name|half_year_number|half_y
ear_name|geo_region_cd|
+-------------+--------------------+------------+-----------+---------+----------------+------------+----------+--------------+------------+----------------+------
--------+-------------+
|   2011-02-20|Sunday, February ...|           2|          4|  Week 04|          201104|           1|  February|             1|          Q1|               1|      
1st Half|           US|
|   2048-03-21|Saturday, March 2...|           1|          8|  Week 08|          204808|           2|     March|             1|          Q1|               1|      
1st Half|           US|
|   2048-03-13|Friday, March 13,...|           7|          6|  Week 06|          204806|           2|     March|             1|          Q1|               1|      
1st Half|           US|
+-------------+--------------------+------------+-----------+---------+----------------+------------+----------+--------------+------------+----------------+------
--------+-------------+
only showing top 3 rows
**********************************************/


spark.read.textFile("/user/edureka_918210/project_retailcart/batchdata/retailcart_department_details.txt").createOrReplaceTempView("retailcart_department_details");

val retailcart_department = 
spark.sql(""" Select 
split(value,'\t')[0] as department_number,
split(value,'\t')[1] as department_category_number,
split(value,'\t')[2] as department_sub_catg_number,
split(value,'\t')[3] as department_description,
split(value,'\t')[4] as department_category_description,
split(value,'\t')[5] as department_sub_catg_desc,
split(value,'\t')[6] as geo_region_cd
from retailcart_department_details """)


/****************************************************************
scala> retailcart_department.printSchema
root
 |-- department_number: string (nullable = true)
 |-- department_category_number: string (nullable = true)
 |-- department_sub_catg_number: string (nullable = true)
 |-- department_description: string (nullable = true)
 |-- department_category_description: string (nullable = true)
 |-- department_sub_catg_desc: string (nullable = true)
 |-- geo_region_cd: string (nullable = true)
scala> 


scala> retailcart_department.show(2)
+-----------------+--------------------------+--------------------------+----------------------+-------------------------------+------------------------+----------
---+
|department_number|department_category_number|department_sub_catg_number|department_description|department_category_description|department_sub_catg_desc|geo_region
_cd|
+-----------------+--------------------------+--------------------------+----------------------+-------------------------------+------------------------+----------
---+
|                1|                      1396|                      2554|  CANDY, TOBACCO, C...|                      HALLOWEEN|              CANDY DISH|          
 US|
|                1|                      1411|                     28737|  CANDY, TOBACCO, C...|                  NON CHOCOLATE|        NON CHOC VARIETY|          
 US|
+-----------------+--------------------------+--------------------------+----------------------+-------------------------------+------------------------+----------
---+
only showing top 2 rows

********************************************************************/

spark.read.textFile("/user/edureka_918210/project_retailcart/batchdata/retailcart_item_details.txt").createOrReplaceTempView("retailcart_item_details");

val retailcart_item = 
spark.sql(""" Select 
split(value,'\t')[0] as item_id,
split(value,'\t')[1] as geo_region_cd,
split(value,'\t')[2] as item_description,
split(value,'\t')[3] as unique_product_cd,
split(value,'\t')[4] as unique_product_cd_desc,
split(value,'\t')[5] as department_number,
split(value,'\t')[6] as department_category_number,
split(value,'\t')[7] as department_sub_catg_number,
split(value,'\t')[8] as vendor_name,
split(value,'\t')[9] as vendor_number,
split(value,'\t')[10] as item_status_cd,
split(value,'\t')[11] as item_status_desc,
split(value,'\t')[12] as unit_cost
from retailcart_item_details """)

spark.read.textFile("/user/edureka_918210/project_retailcart/batchdata/retailcart_store_details.txt").createOrReplaceTempView("retailcart_store_details");

val retailcart_store = 
spark.sql(""" Select 
split(value,'\t')[0] as store_id,
split(value,'\t')[1] as geo_region_cd,
split(value,'\t')[2] as store_name,
split(value,'\t')[3] as sub_division_name,
split(value,'\t')[4] as sub_division_number,
split(value,'\t')[5] as region_number,
split(value,'\t')[6] as region_name,
split(value,'\t')[7] as market_number,
split(value,'\t')[8] as market_name,
split(value,'\t')[9] as city_name,
split(value,'\t')[10] as open_date,
split(value,'\t')[11] as open_status_desc,
split(value,'\t')[12] as postal_cd,
split(value,'\t')[13] as state_prov_cd
from retailcart_store_details """)

spark.read.textFile("/user/edureka_918210/project_retailcart/batchdata/retailcart_currency_details.txt").createOrReplaceTempView("retailcart_currency_details");

val retailcart_currency = 
spark.sql(""" Select 
split(value,'\t')[0] as currency_id,
split(value,'\t')[1] as currency_code,
split(value,'\t')[2] as currency_name,
split(value,'\t')[3] as usd_exchange_rate
from retailcart_currency_details """)

val retailcart_calendar2 = retailcart_calendar.selectExpr("calendar_date", "date_desc", "cast(week_day_nbr as integer) week_day_nbr", "week_number", "week_name", "cast(year_week_number as integer) year_week_number","month_number","month_name","quarter_number","quarter_name","half_year_number","half_year_name","geo_region_cd")
val retailcart_department2 = retailcart_department.selectExpr("cast(department_number as integer) department_number", "cast(department_category_number as integer) department_category_number", "cast(department_sub_catg_number as integer) department_sub_catg_number", "department_description", "department_category_description", "department_sub_catg_desc","geo_region_cd")
val retailcart_item2 = retailcart_item.selectExpr("cast(item_id as integer) item_id", "geo_region_cd", "item_description", "unique_product_cd", "unique_product_cd_desc", "department_number","department_category_number","department_sub_catg_number","vendor_name","vendor_number","item_status_cd","item_status_desc","cast(unit_cost as double) unit_cost")
val retailcart_store2 = retailcart_store.selectExpr("cast(store_id as integer) store_id", "geo_region_cd", "store_name", "sub_division_name", "sub_division_number", "region_number","region_name","market_number","market_name","city_name","open_date","open_status_desc","postal_cd","state_prov_cd")
val retailcart_currency2 = retailcart_currency.selectExpr("cast(currency_id as integer) currency_id", "currency_code", "currency_name", "cast(usd_exchange_rate as double) usd_exchange_rate")

//spark2-shell --jars /mnt/home/edureka_918210/project_retailcart/connector_jars/mysql-connector-java-5.1.48-bin.jar

retailcart_calendar2.write.format("jdbc").option("url","jdbc:mysql://dbserver.edu.cloudlab.com/labuser_database").option("dbtable","retailcart_calendar2").option("user","edu_labuser").option("password","edureka").option("driver","com.mysql.jdbc.Driver").mode("overwrite").save
retailcart_department2.write.format("jdbc").option("url","jdbc:mysql://dbserver.edu.cloudlab.com/labuser_database").option("dbtable","retailcart_department2").option("user","edu_labuser").option("password","edureka").option("driver","com.mysql.jdbc.Driver").mode("overwrite").save
retailcart_item2.write.format("jdbc").option("url","jdbc:mysql://dbserver.edu.cloudlab.com/labuser_database").option("dbtable","retailcart_item2").option("user","edu_labuser").option("password","edureka").option("driver","com.mysql.jdbc.Driver").mode("overwrite").save
retailcart_store2.write.format("jdbc").option("url","jdbc:mysql://dbserver.edu.cloudlab.com/labuser_database").option("dbtable","retailcart_store2").option("user","edu_labuser").option("password","edureka").option("driver","com.mysql.jdbc.Driver").mode("overwrite").save
retailcart_currency2.write.format("jdbc").option("url","jdbc:mysql://dbserver.edu.cloudlab.com/labuser_database").option("dbtable","retailcart_currency2").option("user","edu_labuser").option("password","edureka").option("driver","com.mysql.jdbc.Driver").mode("overwrite").save

//mysql -u edu_labuser -h dbserver.edu.cloudlab.com -p

//password - edureka

CREATE TABLE retailcart_calendar_details SELECT * FROM retailcart_calendar2;
CREATE TABLE retailcart_department_details SELECT * FROM retailcart_department2;
CREATE TABLE retailcart_item_details SELECT * FROM retailcart_item2;
CREATE TABLE retailcart_store_details SELECT * FROM retailcart_store2;
CREATE TABLE retailcart_currency_details SELECT * FROM retailcart_currency2;


val spark = SparkSession.builder().appName("Industry_Grade_Project-spark").master("local[*]").enableHiveSupport().getOrCreate();

val store_item_price_change = spark.read.orc("/bigdatapgp/common_folder/project_retailcart/history_data/store_item_price_change")

val store_item_price_change2 = store_item_price_change.toDF("item_id","store_id","price_chng_activation_ts","geo_region_cd","price_change_reason","business_date","prev_price_amt","curr_price_amt")
store_item_price_change2.write.mode("overwrite").saveAsTable("edureka_918210_DB_Industry_projects.store_item_price_change")

val store_item_sales_data = spark.read.orc("/bigdatapgp/common_folder/project_retailcart/history_data/store_item_sales_data")

val store_item_sales_data2 = store_item_sales_data.toDF("sales_id","Sales_date","store_id","item_id","scan_type","geo_region_cd","currency_code","scan_id","sold_unit_quantity","scan_date","scan_time","scan_dept_nbr")

store_item_sales_data2.write.mode("overwrite").saveAsTable("edureka_918210_DB_Industry_projects.store_item_sales_data")

spark.sql("select * from edureka_918210_DB_Industry_projects.store_item_sales_data limit 10").show()
spark.sql("Show tables in edureka_918210_DB_Industry_projects").show(false)




  }
  
}


    
    
