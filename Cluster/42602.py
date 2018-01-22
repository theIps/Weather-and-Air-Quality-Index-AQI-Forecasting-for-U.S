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

    dict_data['state_code'] = int(words[0])
    dict_data['month'] = int(words[1])
    dict_data['year'] = int(words[2])
    dict_data['observation_count'] = float(words[3])
    dict_data['observation_percent'] = float(words[4])
    dict_data['max_value'] = float(words[5])
    dict_data['max_hour'] = float(words[6])
    dict_data['arithmetic_mean'] = float(words[7])
    dict_data['am_wind'] = float(words[8])
    dict_data['am_temp'] = float(words[9])
    dict_data['am_rh'] = float(words[10])
    dict_data['am_press'] = float(words[11])
    return dict_data

schema = types.StructType([types.StructField('state_code', types.StringType(), True),
                           types.StructField('month', types.StringType(), True),
                           types.StructField('year',types.StringType(), True),
                           types.StructField('observation_count', types.DoubleType(),True),
                           types.StructField('observation_percent', types.DoubleType(), True),
                           types.StructField('max_value', types.DoubleType(), True),
                           types.StructField('max_hour', types.DoubleType(), True),
                           types.StructField('arithmetic_mean', types.DoubleType(), True),
                           types.StructField('am_wind', types.DoubleType(), True),
                           types.StructField('am_temp', types.DoubleType(), True),
                           types.StructField('am_rh', types.DoubleType(), True),
                           types.StructField('am_press', types.DoubleType(), True)])


itr=0
for j in ['42602']:
    train_final={}
    train_final[itr] = spark.createDataFrame(sc.emptyRDD(), schema=schema)

    for year in range (2013, 2018):
        training = spark.read.csv("original/daily_"+ str(j) +"_" + str(year) + ".csv", header = True)
        train = training[['State Code', 'Date Local', 'Observation Count', 'Observation Percent', '1st Max Value', '1st Max Hour','Arithmetic Mean']]


        split_col = functions.split(train['Date Local'], '-')
        train = train.withColumn('Year', split_col.getItem(0))
        train = train.withColumn('Month', split_col.getItem(1))

        train = train.drop('Date Local')
        train.createOrReplaceTempView('train')
        train_g = train.groupBy(train['State Code'],train['Month'],train['Year']).agg(functions.avg(train['Observation Count']).alias('Observation Count'),functions.avg(train['Observation Percent']).alias('Observation Percent'),
           functions.avg(train['1st Max Value']).alias('1st Max Value'),functions.avg(train['1st Max Hour']).alias('1st Max Hour'),
           functions.avg(train['Arithmetic Mean']).alias('Arithmetic Mean'))
        supportl = ['WIND', 'TEMP', 'RH_DP', 'PRESS']
        for i in supportl:

             support = spark.read.csv("support/daily_" + str(i) + "_" + str(year) + ".csv", header=True)
             #support = spark.read.csv("support/daily_" + str(i) + "_" + str(year) + ".csv", header=True)
             support_f = support.select('State Code', 'Date Local', 'Arithmetic Mean')
             split_col = functions.split(support_f['Date Local'], '-')
             support_f = support_f.withColumn('Year', split_col.getItem(0))
             support_f = support_f.withColumn('Month', split_col.getItem(1))
             support_f = support_f.drop('Date Local')
             support_t = support_f.groupBy([support_f['State Code'],support_f['Month'],support_f['Year']]).agg(
                         functions.avg(support_f['Arithmetic Mean']).alias('AM'))
             support_g = support_t.select(support_t['State Code'].alias('sc'),support_t['Month'].alias('m'),support_t['Year'].alias('y'),support_t['AM'])
             train_g = train_g.join(support_g,[(train_g['State Code'] == support_g['sc']) & (train_g['Year'] == support_g['y']) & (train_g['Month']== support_g['m'])
                                    ]).drop('sc','m','y').select('*').sort('State Code','Year','Month')

        train_final[itr] = train_final[itr].union(train_g)

    table_name = "g" + str(j)
    rdd = train_final[itr].rdd.map(tuple)
    words = rdd.map(load_table)
    words.saveToCassandra(keyspace, table_name)

    itr += 1





