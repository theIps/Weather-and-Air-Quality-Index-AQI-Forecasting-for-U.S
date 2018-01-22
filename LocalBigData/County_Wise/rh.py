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

schema = types.StructType([types.StructField('county_code', types.IntegerType(), True),
                           types.StructField('month', types.IntegerType(), True),
                           types.StructField('year', types.IntegerType(), True),
                           types.StructField('am_rh', types.DoubleType(), True)])


train_final = spark.createDataFrame(sc.emptyRDD(), schema=schema)

for year in range(2013, 2018):
    support = spark.read.csv("/home/ldua/Desktop/BigDataProject/support/daily_RH_DP_" + str(year) + ".csv", header=True)

    support_f = support.select('County Code', 'Date Local', 'Arithmetic Mean').where(support['State Code'] == 6)
    split_col = functions.split(support_f['Date Local'], '-')
    support_f = support_f.withColumn('Year', split_col.getItem(0))
    support_f = support_f.withColumn('Month', split_col.getItem(1))
    support_f = support_f.drop('Date Local')
    support_t = support_f.groupBy([support_f['County Code'], support_f['Month'], support_f['Year']]).agg(
        functions.avg(support_f['Arithmetic Mean']).alias('AM'))
    support_g = support_t.select(support_t['County Code'].alias('sc'), support_t['Month'].alias('m'),
                                 support_t['Year'].alias('y'), support_t['AM'])

    train_final = train_final.union(support_g)

train_final.show()
train_final = train_final.withColumn('county_code', train_final['county_code'].cast('Integer'))
train_final = train_final.withColumn('month', train_final['month'].cast('Integer'))
train_final = train_final.withColumn('year', train_final['year'].cast('Integer'))
train_final = train_final.withColumn('am_rh', train_final['am_rh'].cast('Integer'))

schema2 = types.StructType([types.StructField('county_code', types.IntegerType(), True),
                            types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True)])

scheme3 = types.StructType([types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True),
                            types.StructField('predicted_rh', types.IntegerType(), True),
                            types.StructField('county_code', types.IntegerType(), True)
                            ])
testing = spark.read.csv("/home/ldua/Desktop/County/county_test.csv", schema=schema2)

training = train_final

state = training.select('county_code').distinct()
stateval = state.collect()
training.createOrReplaceTempView("training")
testing.createOrReplaceTempView("testing")
predictions = spark.createDataFrame(sc.emptyRDD(), schema=scheme3)
for i in stateval:
    print(i['county_code'])
    df = spark.sql('''select month,year,am_rh from training where county_code=''' + str(
        i["county_code"]) + ''' order by year LIMIT 60''')

    df_test = spark.sql('''select month, year from testing where county_code=''' + str(i["county_code"]))


    prediction_Col_Name = "predicted_rh"

    vecAssembler = VectorAssembler(inputCols=["month", "year"], outputCol="features")
    # lr = LinearRegression(featuresCol="features", labelCol="am_rh", predictionCol=prediction_Col_Name)
    # rfr = RandomForestRegressor(featuresCol="features", labelCol="am_rh", predictionCol=prediction_Col_Name)
    # dtr = DecisionTreeRegressor(featuresCol="features", labelCol="am_rh", predictionCol=prediction_Col_Name)
    gbtr = GBTRegressor(featuresCol="features", labelCol="am_rh", predictionCol=prediction_Col_Name)

    # Linear_Regressor = [vecAssembler, lr]
    # Random_Forest = [vecAssembler, rfr]
    # DecisionTree_Regressor = [vecAssembler, dtr]
    GBT_Regressor = [vecAssembler, gbtr]

    models = [
        # ('Linear Regressor', Pipeline(stages=Linear_Regressor)),
        # ('Random Forest Regressor', Pipeline(stages=Random_Forest)),
        # ('Decision Tree Regressor', Pipeline(stages=DecisionTree_Regressor)),
        ('GBT Regressor', Pipeline(stages=GBT_Regressor)),
    ]

    # evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="am_rh", metricName="mse")

    # split = df.randomSplit([0.80, 0.20])
    # train = split[0]
    # test = split[1]
    # train = train.cache()
    # test = test.cache()

    for label, pipeline in models:
        model = pipeline.fit(df)

        pred = model.transform(df_test)
        pred = pred.drop("features")
        pred = pred.withColumn('county_code', functions.lit(i["county_code"]))
        predictions = predictions.union(pred)


predictions.coalesce(1).write.csv('/home/ldua/Desktop/County/Output/predicted_rh', sep=',', header=True)