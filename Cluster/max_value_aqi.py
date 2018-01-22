import sys, os
#import pyspark_cassandra
from pyspark import SparkConf
from pyspark.sql import SparkSession, types, functions
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor, DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark_cassandra
from cassandra.cluster import Cluster

from cassandra.query import BatchStatement, SimpleStatement

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
        .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

sc = pyspark_cassandra.CassandraSparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
assert sc.version >= '2.2'  # make sure we have Spark 2.2+
keyspace = 'ldua'

def df_for(keyspace, table, split_size=100):
    df = spark.createDataFrame(sc.cassandraTable(keyspace, table, split_size=split_size).setName(table))
    df.createOrReplaceTempView(table)
    return df

def load_table(words):
    dict_data = {}
    dict_data['month'] = int(words[0])
    dict_data['year'] = int(words[1])
    dict_data['am_temp'] = float(words[2])
    dict_data['am_press'] = float(words[3])
    dict_data['am_wind'] = float(words[4])
    dict_data['am_rh'] = float(words[5])
    dict_data['max_value_pred_' + criteria_gas] = float(words[6])
    dict_data['state_code'] = int(words[7])
    return dict_data

predicted_temp = df_for(keyspace, 'predicted_temp', split_size=None)
predicted_pressure = df_for(keyspace, 'predicted_press', split_size=None)
predicted_wind = df_for(keyspace, 'predicted_wind', split_size=None)
predicted_rh = df_for(keyspace, 'predicted_rh', split_size=None)

schema_test = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                                types.StructField('month', types.IntegerType(), True),
                                types.StructField('year', types.IntegerType(), True)])

test = spark.read.csv("test.csv", header=True, schema=schema_test)
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

support = support.fillna({'am_press': 0, 'am_temp': 0, 'am_wind': 0, 'am_rh': 0})
support.show()

support.createOrReplaceTempView('support')

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

for criteria_gas in ['44201', '42401', '42101', '42602', '88101','88502']:
    j = 0
    itr = 0

    table_name = "g" + str(criteria_gas)

    schema2 = types.StructType([types.StructField('month', types.IntegerType(), True),
                                types.StructField('year', types.IntegerType(), True),
                                types.StructField('am_temp', types.DoubleType(), True),
                                types.StructField('am_press', types.DoubleType(), True),
                                types.StructField('am_wind', types.DoubleType(), True),
                                types.StructField('am_rh', types.DoubleType(), True),
                                types.StructField('max_value_pred_' + criteria_gas, types.IntegerType(), True),
                                types.StructField('state_code', types.IntegerType(), True)])

    predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=schema2)

    training = df_for(keyspace, table_name, split_size=None)

    state = training.select('state_code').distinct()
    stateval = state.collect()
    training.createOrReplaceTempView('training')
    print(stateval)

    for i in stateval:
        print(i['state_code'])
        itr += 1
        df = spark.sql(
            '''select month,year,am_temp,am_press,am_wind,am_rh,max_value from training where state_code=''' + str(
                i['state_code']))  # + ''' and year>=2015''')

        df_test = spark.sql(
            '''select month,year,am_temp,am_press,am_wind,am_rh from testing where state_code=''' + str(
                i['state_code']))

        prediction_Col_Name = "max_value_pred_" + str(criteria_gas)

        vecAssembler = VectorAssembler(inputCols=["month", "year", "am_temp", "am_press", "am_wind", "am_rh"],
                                       outputCol="features")
        # lr = LinearRegression(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        # rfr = RandomForestRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        # dtr = DecisionTreeRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)
        gbtr = GBTRegressor(featuresCol="features", labelCol="max_value", predictionCol=prediction_Col_Name)

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

        evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="max_value",
                                        metricName="mse")

        # split = df.randomSplit([0.80, 0.20])
        # train = split[0]
        # test = split[1]
        # train = train.cache()
        # test = test.cache()

        # min = 1000
        # for label, pipeline in models:
        #     model = pipeline.fit(df)
        #     pred = model.transform(df)
        #     score = evaluator.evaluate(pred)d
        #     # print("\nCriteria Gas", criteria_gas)
        #     print(label, score)
        #     if min > score:
        #         min = score
        #         min_pipe = pipeline
        # print("\n----Criteria Gas-----", criteria_gas)
        # print(min_pipe, min)

        for label, pipeline in models:
            model = pipeline.fit(df)
            pred = model.transform(df_test)
            pred = pred.drop("features")
            pred = pred.withColumn('state_code', functions.lit(i["state_code"]))
            predictions[criteria_gas] = predictions[criteria_gas].union(pred)

        if (itr == 10):
            rdd = predictions[criteria_gas].rdd.map(tuple)
            words = rdd.map(load_table)
            words.saveToCassandra(keyspace, "predicted_aqi_" + criteria_gas)

            predictions[criteria_gas] = spark.createDataFrame(sc.emptyRDD(), schema=schema2)
            itr = 0
            j += 1

    rdd = predictions[criteria_gas].rdd.map(tuple)
    words = rdd.map(load_table)
    words.saveToCassandra(keyspace, "predicted_aqi_"+ criteria_gas)

predictions[criteria_gas].show()