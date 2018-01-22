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

# def df_for(keyspace, table, split_size=100):
#     df = spark.createDataFrame(sc.cassandraTable(keyspace, table, split_size=split_size).setName(table))
#     df.createOrReplaceTempView(table)
#     return df

# Table1 = df_for(keyspace, 'predicted_temp', split_size=None)
# Table2 = df_for(keyspace, 'predicted_pressure', split_size=None)
# Table3 = df_for(keyspace, 'predicted_wind', split_size=None)
# Table4 = df_for(keyspace, 'predicted_rh', split_size=None)

train_f = {}
for j in ['44201', '42401', '42101', '42602','88502','88101']:
    schema = types.StructType([
        types.StructField('month', types.IntegerType(), True),
        types.StructField('year', types.IntegerType(), True),
        types.StructField('am_temp', types.DoubleType(), True),
        types.StructField('am_press', types.DoubleType(), True),
        types.StructField('am_wind', types.DoubleType(), True),
        types.StructField('am_rh', types.DoubleType(), True),
        types.StructField('max_value_pred_' + str(j), types.StringType(), True),
        types.StructField('county_code', types.StringType(), True)])

    train_f[j] = spark.createDataFrame(sc.emptyRDD(), schema=schema)

    for i in ['0', '1']:
        if (j=='42401' and i=='1'):
            continue
        else:
            training = spark.read.csv(
                "/home/ldua/Desktop/County/max_value/predicted_max_value_" + str(j) + "_" + str(i) + "/" + str(
                    i) + ".csv", header=True, schema=schema)
            train_f[j] = train_f[j].union(training)
            print(j, i)

schema2 = types.StructType([types.StructField('county_code', types.IntegerType(), True),
                            types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True)])

test = spark.read.csv("/home/ldua/Desktop/County/county_test.csv", header=True, schema=schema2)

train_f['44201'].createOrReplaceTempView('g44201')
train_f['42401'].createOrReplaceTempView('g42401')
train_f['42101'].createOrReplaceTempView('g42101')
train_f['42602'].createOrReplaceTempView('g42602')
train_f['88502'].createOrReplaceTempView('g88502')
train_f['88101'].createOrReplaceTempView('g88101')
test.createOrReplaceTempView('test')

max_value = spark.sql('''SELECT t.county_code, t.month, t.year, a.max_value_pred_44201, b.max_value_pred_42401, c.max_value_pred_42101, d.max_value_pred_42602, e.max_value_pred_88502, f.max_value_pred_88101
FROM test t
FULL OUTER JOIN g44201 a
ON t.county_code=a.county_code AND t.month=a.month AND t.year=a.year
FULL OUTER JOIN g42401 b
ON t.county_code=b.county_code AND t.month=b.month AND t.year=b.year
FULL OUTER JOIN g42101 c
ON t.county_code=c.county_code AND t.month=c.month AND t.year=c.year
FULL OUTER JOIN g42602 d
ON t.county_code=d.county_code AND t.month=d.month AND t.year=d.year
FULL OUTER JOIN g88502 e
ON t.county_code=e.county_code AND t.month=e.month AND t.year=e.year
FULL OUTER JOIN g88101 f
ON t.county_code=f.county_code AND t.month=f.month AND t.year=f.year''')
max_value=max_value.fillna({'max_value_pred_44201':0,'max_value_pred_42401':0,'max_value_pred_42101':0,'max_value_pred_42602':0,'max_value_pred_88502':0,'max_value_pred_88101':0})

max_value.coalesce(1).write.csv('/home/ldua/Desktop/County/max_value_combined/', sep=',', header=True)




