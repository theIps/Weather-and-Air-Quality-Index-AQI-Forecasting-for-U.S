import sys, os
from pyspark.sql import SparkSession, types, functions
from pyspark import SparkConf, SparkContext
import sys, re, datetime, uuid

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

cluster_seeds = ['199.60.17.171', '199.60.17.188']

cluster_seeds = ['199.60.17.171', '199.60.17.188']

conf = SparkConf().setAppName('example code') \
    .set('spark.cassandra.connection.host', ','.join(cluster_seeds))

spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
sc = spark.sparkContext
assert sys.version_info >= (3, 4)  # make sure we have Python 3.4+
assert spark.version >= '2.2'  # make sure we have Spark 2.2+



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
for j in ['44201']:
    train_final={}
    train_final[itr] = spark.createDataFrame(sc.emptyRDD(), schema=schema)

    for year in range (1998, 2018):
        training = spark.read.csv("/home/ldua/Desktop/BigDataProject/original/daily_"+ str(j) +"_" + str(year) + ".csv", header = True)
        #training = spark.read.csv("original/daily_" + str(j) + "_" + str(year) + ".csv", header=True)

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

             support = spark.read.csv("/home/ldua/Desktop/BigDataProject/support/daily_" + str(i) + "_" + str(year) + ".csv", header=True)
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



    train_final[itr].coalesce(1).write.csv('/home/ldua/Desktop/FinalBig/FinalOutput/' + str(j), sep=',', header=True)


    itr += 1





