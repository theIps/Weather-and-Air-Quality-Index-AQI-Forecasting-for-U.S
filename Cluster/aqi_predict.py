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

#output = sys.argv[1]
keyspace = 'ldua'

def df_for(keyspace, table, split_size=100):
    df = spark.createDataFrame(sc.cassandraTable(keyspace, table, split_size=split_size).setName(table))
    df.createOrReplaceTempView(table)
    return df

schema2 = types.StructType([types.StructField('state_code', types.IntegerType(), True),
                            types.StructField('month', types.IntegerType(), True),
                            types.StructField('year', types.IntegerType(), True)])

test = spark.read.csv("test.csv", header=True, schema=schema2)
test.createOrReplaceTempView('test')

predicted_aqi_44201 = df_for(keyspace, 'predicted_aqi_44201', split_size=None)
predicted_aqi_42101 = df_for(keyspace, 'predicted_aqi_42101', split_size=None)
predicted_aqi_42401 = df_for(keyspace, 'predicted_aqi_42401', split_size=None)
predicted_aqi_42602 = df_for(keyspace, 'predicted_aqi_42602', split_size=None)
predicted_aqi_88502 = df_for(keyspace, 'predicted_aqi_88502', split_size=None)
predicted_aqi_88101 = df_for(keyspace, 'predicted_aqi_88101', split_size=None)

max_value = spark.sql('''SELECT t.state_code, t.month, t.year, a.max_value_pred_44201, b.max_value_pred_42401, c.max_value_pred_42101, d.max_value_pred_42602, e.max_value_pred_88502, f.max_value_pred_88101
FROM test t
FULL OUTER JOIN predicted_aqi_44201 a
ON t.state_code=a.state_code AND t.month=a.month AND t.year=a.year
FULL OUTER JOIN predicted_aqi_42401 b
ON t.state_code=b.state_code AND t.month=b.month AND t.year=b.year
FULL OUTER JOIN predicted_aqi_42101 c
ON t.state_code=c.state_code AND t.month=c.month AND t.year=c.year
FULL OUTER JOIN predicted_aqi_42602 d
ON t.state_code=d.state_code AND t.month=d.month AND t.year=d.year
FULL OUTER JOIN predicted_aqi_88502 e
ON t.state_code=e.state_code AND t.month=e.month AND t.year=e.year
FULL OUTER JOIN predicted_aqi_88101 f
ON t.state_code=f.state_code AND t.month=f.month AND t.year=f.year''')
max_value=max_value.fillna({'max_value_pred_44201':0,'max_value_pred_42401':0,'max_value_pred_42101':0,'max_value_pred_42602':0,'max_value_pred_88502':0,'max_value_pred_88101':0})

def load_table(words):
   dict_data = {}
   dict_data['state_code'] = int(words[0])
   dict_data['month'] = int(words[1])
   dict_data['year'] = int(words[2])
   dict_data['max_value_pred_44201'] = float(words[3])
   dict_data['max_value_pred_42401'] = float(words[4])
   dict_data['max_value_pred_42101'] = float(words[5])
   dict_data['max_value_pred_42602'] = float(words[6])
   dict_data['max_value_pred_88502'] = float(words[7])
   dict_data['max_value_pred_88101'] = float(words[8])
   dict_data['gloabl_aqi'] = float(words[9])
   return dict_data

def aqi_cal(val,aqilevel,gaslevel):
    length = len(gaslevel)

    for i in range(len(gaslevel)):
        if (val < gaslevel[i]):
            num1 = val - gaslevel[i - 1]
            num2 = aqilevel[i] - aqilevel[i - 1]
            den = gaslevel[i] - gaslevel[i - 1]
            aqival = ((num1 * num2) / den) + aqilevel[i - 1]
            break
        else:
            if (val >= gaslevel[length-1]):
                aqival = aqilevel[length-1]-1
                break

    return aqival

# def aqi_so2(val):
#
#      return aqi


def transform(line):
    #val = line.split(',')
    #if val[0] == 'state_code':
        #return (val[0],val[1],val[2],val[3],val[4],val[5],val[6],val[7],val[8],'global_aqi')
        #return line+',Global_AQI'

    #else:
    aqi_level = [0,51,101,151,201,301,401,500]
    ozone_level = [0,.055,.071,.086,.106,.201]
    so_level = [0,36,76,186,305,605,805,1005]
    co_level = [0,4.5,9.5,12.5,15.5,30.5,40.5,50.5]
    no_level = [0,54,101,361,650,1250,1650,2050]
    pm_level = [0,12.1,35.5,55.5,150.5,250.5,350.5,500.5]
    aqi_oz = aqi_cal(float(line.max_value_pred_44201),aqi_level,ozone_level)
    aqi_so = aqi_cal(float(line.max_value_pred_42401), aqi_level, so_level)
    aqi_co = aqi_cal(float(line.max_value_pred_42101), aqi_level, co_level)
    aqi_no = aqi_cal(float(line.max_value_pred_42602), aqi_level, no_level)
    aqi_pma = aqi_cal(float(line.max_value_pred_88502), aqi_level, pm_level)
    aqi_pmb = aqi_cal(float(line.max_value_pred_88101), aqi_level, pm_level)


        # val[3] = float(val[3])
        # val[4] = float(val[4])
        # val[5] = float(val[5])
        # val[6] = float(val[6])
        # for i in range(len(ozone_level)):
        #     if(val[3]< ozone_level[i]):
        #         num1 = val[3] - ozone_level[i-1]
        #         num2 = aqi_level[i] - aqi_level[i-1]
        #         den = ozone_level[i] - ozone_level[i-1]
        #         aqi_oz = ((num1 * num2)/den)+aqi_level[i-1]
        #         break
        #     else:
        #         if(val[3] >= ozone_level[5]):
        #             aqi_oz = 300
        #             break
        #
        # for i in range(len(so_level)):
        #     if (val[4] < so_level[i]):
        #         num1 = val[4] - so_level[i - 1]
        #         num2 = aqi_level[i] - aqi_level[i - 1]
        #         den = so_level[i] - so_level[i - 1]
        #         aqi_so = ((num1 * num2) / den) + aqi_level[i - 1]
        #         break
        #     else:
        #         if (val[4] > so_level[7]):
        #             aqi_so = 500
        #             break
        #
        # for i in range(len(co_level)):
        #     if (val[5] < co_level[i]):
        #         num1 = val[5] - co_level[i - 1]
        #         num2 = aqi_level[i] - aqi_level[i - 1]
        #         den = co_level[i] - co_level[i - 1]
        #         aqi_co = ((num1 * num2) / den) + aqi_level[i - 1]
        #         break
        #     else:
        #         if (val[5] > co_level[7]):
        #             aqi_co = 500
        #             break
        #
        # for i in range(len(no_level)):
        #     if (val[6] < no_level[i]):
        #         num1 = val[6] - no_level[i - 1]
        #         num2 = aqi_level[i] - aqi_level[i - 1]
        #         den = no_level[i] - no_level[i - 1]
        #         aqi_no = ((num1 * num2) / den) + aqi_level[i - 1]
        #         break
        #     else:
        #         if (val[6] > no_level[7]):
        #             aqi_no = 500
        #             break

    glo = [aqi_no,aqi_so,aqi_oz,aqi_co,aqi_pma,aqi_pmb]

    return (line.state_code,line.month,line.year,aqi_oz,aqi_so,aqi_co,aqi_no,aqi_pma,aqi_pmb,max(glo))

# explicit_schema = types.StructType([types.StructField('State Code', types.IntegerType(), True),
#                    types.StructField('Month', types.IntegerType(), True),
#                    types.StructField('Year',types.IntegerType(), True),
#                    types.StructField('AM_Predicted_44201', types.DoubleType(), True),
#                    types.StructField('AM_Predicted_42401', types.DoubleType(), True)])

#State Code,Year,Month,AM_Predicted_44201,AM_Predicted_42401
#Row(State Code=1, Month=2011, Year=1, AM_Predicted_44201=0.02665985549600323, AM_Predicted_42401=1.6022149730848756)
#training = sc.textFile("/home/ldua/Desktop/BigDataProject/Output/AQI/part-00000-e88f6806-9bdc-4906-84f7-0647e9a022d8-c000.csv")
#training = spark.read.csv("/home/ldua/Desktop/BigDataProject/Output/AQI/part-00000-e88f6806-9bdc-4906-84f7-0647e9a022d8-c000.csv", header= True, schema= explicit_schema)
#aqi = training.map(transform)

training = max_value.rdd
aqi = training.map(transform)
#aqi.saveAsTextFile(output)
# header = aqi.first()
# data = aqi.filter(lambda row : row != header).toDF(header)
words = aqi.map(load_table)
words.saveToCassandra(keyspace, 'predicted_global_aqi')