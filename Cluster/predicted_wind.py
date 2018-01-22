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

def load_table(words):
   dict_data = {}
   dict_data['month'] = int(words[0])
   dict_data['year'] = int(words[1])
   dict_data['predicted_wind'] = float(words[2])
   dict_data['state_code'] = int(words[3])
   return dict_data


def df_for(keyspace, table, split_size=100):
    df = spark.createDataFrame(sc.cassandraTable(keyspace, table, split_size=split_size).setName(table))
    df.createOrReplaceTempView(table)
    return df


schema2 = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                          types.StructField('month', types.IntegerType(), True),
                          types.StructField('year',types.IntegerType(), True)])

testing = spark.read.csv("test.csv", header=True, schema=schema2)

predictions = {}
i = 0
schema = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                           types.StructField('month', types.IntegerType(), True),
                           types.StructField('year', types.IntegerType(), True),
                           types.StructField('am_wind', types.DoubleType(), True)])


train_final = spark.createDataFrame(sc.emptyRDD(), schema=schema)

for year in range(2013, 2018):
    support = spark.read.csv("support/daily_WIND_" + str(year) + ".csv", header=True)

    support_f = support.select('State Code', 'Date Local', 'Arithmetic Mean')
    split_col = functions.split(support_f['Date Local'], '-')
    support_f = support_f.withColumn('Year', split_col.getItem(0))
    support_f = support_f.withColumn('Month', split_col.getItem(1))
    support_f = support_f.drop('Date Local')
    support_t = support_f.groupBy([support_f['State Code'], support_f['Month'], support_f['Year']]).agg(
        functions.avg(support_f['Arithmetic Mean']).alias('AM'))
    support_g = support_t.select(support_t['State Code'].alias('sc'), support_t['Month'].alias('m'),
                                 support_t['Year'].alias('y'), support_t['AM'])
    # train_g = train_g.join(support_g,[(train_g['State Code'] == support_g['sc']) & (train_g['Year'] == support_g['y']) & (train_g['Month']== support_g['m'])
    #                         ]).drop('sc','m','y').select('*').sort('State Code','Year','Month')

    train_final = train_final.union(support_g)

train_final.show()
train_final = train_final.withColumn('state_code', train_final['state_code'].cast('Integer'))
train_final = train_final.withColumn('month', train_final['month'].cast('Integer'))
train_final = train_final.withColumn('year', train_final['year'].cast('Integer'))
train_final = train_final.withColumn('am_wind', train_final['am_wind'].cast('Integer'))

schema2 = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                            types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True)])

scheme3 = types.StructType([types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True),
                            types.StructField('predicted_wind', types.IntegerType(), True),
                            types.StructField('state_code', types.IntegerType(), True)
                            ])
testing = spark.read.csv("test.csv", schema=schema2)

training = train_final

state = training.select('state_code').distinct()
stateval = state.collect()
training.createOrReplaceTempView("training")
testing.createOrReplaceTempView("testing")
predictions = spark.createDataFrame(sc.emptyRDD(), schema=scheme3)
for i in stateval:
    print(i['state_code'])
    df = spark.sql('''select month,year,am_wind from training where state_code=''' + str(
        i["state_code"]) + ''' order by year LIMIT 60''')

    df_test = spark.sql('''select month, year from testing where state_code=''' + str(i["state_code"]))
    df_test = testing.select('month', 'year').where((testing['state_code'] == i["state_code"]))

    prediction_Col_Name = "predicted_wind"

    vecAssembler = VectorAssembler(inputCols=["month", "year"], outputCol="features")
    # lr = LinearRegression(featuresCol="features", labelCol="am_wind", predictionCol=prediction_Col_Name)
    # rfr = RandomForestRegressor(featuresCol="features", labelCol="am_wind", predictionCol=prediction_Col_Name)
    # dtr = DecisionTreeRegressor(featuresCol="features", labelCol="am_wind", predictionCol=prediction_Col_Name)
    gbtr = GBTRegressor(featuresCol="features", labelCol="am_wind", predictionCol=prediction_Col_Name)

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

    # evaluator = RegressionEvaluator(predictionCol=prediction_Col_Name, labelCol="am_wind", metricName="mse")

    # split = df.randomSplit([0.80, 0.20])
    # train = split[0]
    # test = split[1]
    # train = train.cache()
    # test = test.cache()

    for label, pipeline in models:
        model = pipeline.fit(df)

        pred = model.transform(df_test)
        pred = pred.drop("features")
        pred = pred.withColumn('state_code', functions.lit(i["state_code"]))
        predictions = predictions.union(pred)


# predictions.coalesce(1).write.csv('/home/ldua/Desktop/Final/predicted_wind', sep=',', header=True)
# df_final = train_final.union(predictions)
# df_final.coalesce(1).write.csv('/home/ldua/Desktop/Final/predicted_wind_i', sep=',', header=True)

rdd = predictions.rdd.map(tuple)
words = rdd.map(load_table)
words.saveToCassandra(keyspace, 'predicted_wind')