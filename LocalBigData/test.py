import sys, os
from pyspark.sql import SparkSession, types, functions
from pyspark import SparkConf, SparkContext
import sys
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

os.environ["PYSPARK_PYTHON"] = "python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
   .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+



predicted_temp = spark.read.csv("/home/ldua/Desktop/FinalBig/Output/predicted_temp/predicted_temp.csv", header=True)
predicted_temp.createOrReplaceTempView('predicted_temp')
predicted_pressure = spark.read.csv("/home/ldua/Desktop/FinalBig/Output/predicted_press/predicted_press.csv",
                                   header=True)
predicted_pressure.createOrReplaceTempView('predicted_press')
predicted_wind = spark.read.csv("/home/ldua/Desktop/FinalBig/Output/predicted_wind/predicted_wind.csv", header=True)
predicted_wind.createOrReplaceTempView('predicted_wind')
predicted_rh = spark.read.csv("/home/ldua/Desktop/FinalBig/Output/predicted_rh/predicted_rh.csv", header=True)
predicted_rh.createOrReplaceTempView('predicted_rh')



schema_test = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                            types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True)])

test = spark.read.csv("/home/ldua/Desktop/BigDataProject/test.csv", header=True, schema=schema_test)
test.createOrReplaceTempView("test")

support = spark.sql('''SELECT t.state_code, t.month, t.year, a.predicted_temp as am_temp, b.predicted_press as am_press, c.predicted_wind as am_wind, d.predicted_rh as am_rh
From test t
Full Outer JOIN predicted_temp a
ON t.state_code=a.state_code AND t.month=a.month AND t.year=a.year
Full Outer JOIN predicted_press b
ON t.state_code=b.state_code AND t.month=b.month AND t.year=b.year
Full Outer JOIN predicted_wind c
ON t.state_code=c.state_code AND t.month=c.month AND t.year=c.year
Full Outer JOIN predicted_rh d
ON t.state_code=d.state_code AND t.month=d.month AND t.year=d.year''')

support.createOrReplaceTempView('support')

# support.show()

schema = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                          types.StructField('month', types.IntegerType(), True),
                          types.StructField('year', types.IntegerType(), True),
                          types.StructField('observation_count', types.DoubleType(), True),
                          types.StructField('observation_percent', types.DoubleType(), True),
                          types.StructField('max_value', types.DoubleType(), True),
                          types.StructField('max_hour', types.DoubleType(), True),
                          types.StructField('arithmetic_mean', types.DoubleType(), True),
                          types.StructField('am_wind', types.DoubleType(), True),
                          types.StructField('am_temp', types.DoubleType(), True),
                          types.StructField('am_rh', types.DoubleType(), True),
                          types.StructField('am_press', types.DoubleType(), True)])
#

support = support.withColumn('state_code', support['state_code'].cast('Integer'))
support = support.withColumn('month', support['month'].cast('Integer'))
support = support.withColumn('year', support['year'].cast('Integer'))
support = support.withColumn('am_temp', support['am_temp'].cast('Double'))
support = support.withColumn('am_press', support['am_press'].cast('Double'))
support = support.withColumn('am_wind', support['am_wind'].cast('Double'))
support = support.withColumn('am_rh', support['am_rh'].cast('Double'))

testing = support
testing.createOrReplaceTempView('testing')

predictions = {}
i = 0

for criteria_gas in ['44201','42401','42101','42602']:
   j = 0
   itr = 0
   schema2 = types.StructType([types.StructField('month', types.IntegerType(), True),
                               types.StructField('year', types.IntegerType(), True),
                               types.StructField('am_temp', types.DoubleType(), True),
                               types.StructField('am_press', types.DoubleType(), True),
                               types.StructField('am_wind', types.DoubleType(), True),
                               types.StructField('am_rh', types.DoubleType(), True),
                               types.StructField('max_value_pred_' + criteria_gas, types.IntegerType(), True),
                               types.StructField('state_code', types.IntegerType(), True)])

   predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=schema2)

   training = spark.read.csv(
       "/home/ldua/Desktop/FinalBig/FinalOutput/" + str(criteria_gas) + "/" + str(criteria_gas) + ".csv", header=True,
       schema=schema)


   state = training.select('state_code').distinct()  #.where(training['year']>=2015)
   stateval = state.collect()
   training.createOrReplaceTempView('training')
   print(stateval)
   # print(monthval)

   for i in stateval:
       print(i['state_code'])
       itr+=1
       df = spark.sql(
           '''select month,year,am_temp,am_press,am_wind,am_rh,max_value from training where state_code=4''') #+ ''' and year>=2015''')

       df_test = spark.sql('''select month,year,am_temp,am_press,am_wind,am_rh from testing where state_code=4''')

       df.show()
       df_test.show()




#        prediction_Col_Name = "max_value_pred_" + str(criteria_gas)
#
#        vecAssembler = VectorAssembler(inputCols=["month", "year", "am_temp", "am_press", "am_wind", "am_rh"],
#                                       outputCol="features")
#        # lr = LinearRegression(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
#        # rfr = RandomForestRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
#        # dtr = DecisionTreeRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
#        gbtr = GBTRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
#
#        # Linear_Regressor = [vecAssembler, lr]
#        # Random_Forest = [vecAssembler, rfr]
#        # DecisionTree_Regressor = [vecAssembler, dtr]
#        GBT_Regressor = [vecAssembler, gbtr]
#
#        models = [
#            # ('Linear Regressor', Pipeline(stages=Linear_Regressor)),
#            # ('Random Forest Regressor', Pipeline(stages=Random_Forest)),
#            # ('Decision Tree Regressor', Pipeline(stages=DecisionTree_Regressor)),
#            ('GBT Regressor', Pipeline(stages=GBT_Regressor)),
#        ]
#
#        evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="max_value",
#                                        metricName="mse")
#
#        # split = df.randomSplit([0.80, 0.20])
#        # train = split[0]
#        # test = split[1]
#        # train = train.cache()
#        # test = test.cache()
#
#        # min = 1000
#        # for label, pipeline in models:
#        #     model = pipeline.fit(df)
#        #     pred = model.transform(df)
#        #     score = evaluator.evaluate(pred)
#        #     # print("\nCriteria Gas", criteria_gas)
#        #     print(label, score)
#        #     if min > score:
#        #         min = score
#        #         min_pipe = pipeline
#        # print("\n----Criteria Gas-----", criteria_gas)
#        # print(min_pipe, min)
#
#        for label, pipeline in models:
#            model = pipeline.fit(df)
#            pred = model.transform(df_test)
#            pred = pred.drop("features")  # ,"am_temp", "am_press", "am_wind", "am_rh")
#            pred = pred.withColumn('state_code', functions.lit(i["state_code"]))
#            # pred.show()
#            predictions[criteria_gas] = predictions[criteria_gas].union(pred)
#            # predictions[criteria_gas].show()
#        if(itr==30):
#
#            predictions[criteria_gas].coalesce(1).write.csv(
#                '/home/ldua/Desktop/FinalBig/max_value/predicted_max_value_'+str(criteria_gas)+'_'+str(j), sep=',',
#                header=True)
#            predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=schema2)
#            itr = 0
#            j += 1
#
#    predictions[criteria_gas].coalesce(1).write.csv(
#            '/home/ldua/Desktop/FinalBig/max_value/predicted_max_value_'+str(criteria_gas)+'_'+str(j), sep=',',
#            header=True)
#
#
#
#
# predictions[criteria_gas].show()
#
#
#
