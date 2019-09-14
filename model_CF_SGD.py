from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType

RATIING_DATASET = "animelists_cleaned.csv"
USER_DATASET = "users_cleaned.csv"

UserAnimeSchema = StructType([
    StructField("username", StringType(), True),
    StructField("animeId", IntegerType(), True),
    StructField("watched_episodes", LongType(), True),
    StructField("start_date", StringType(), True),
    StructField("finish_date", StringType(), True),  
    StructField("rating", IntegerType(), True),
    StructField("status", IntegerType(), True),
    StructField("rewatching", StringType(), True),
    StructField("rewatching_ep", LongType(), True),
    StructField("last_updated_date", StringType(), True),
    StructField("tags", StringType(), True),
])

user_anime = spark.read\
                  .schema(UserAnimeSchema)\
                  .option("header", "false")\
                  .option("mode", "DROPMALFORMED")\
                  .csv(DATASET)

user_anime_cleaned = user_anime.select("username", "animeId", "rating")\
                               .na.drop(subset = ["username", "animeId"])\
                               .filter("rating <= 10 and rating >= 1")
# Preprocess the rating data

users = spark.read.csv(USER_DATASET, header = "true", inferSchema="true")\
             .select("username", "user_id")\
             .na.drop()
# Preprocess the user data

ratings = user_anime_cleaned.join(users, "username")\
                            .select("user_id", "animeId", "rating")\
                            .withColumnRenamed("user_id", "userId")
# three columns "userId", "animeId", "rating" all in integers and nullable
# pure ratings without biases

# adding baseline predictors
# rui = u + bu + bi
# parallel SGD implementation
# first only use the iterations to control and show the rmse as well as accuracy simultaneously
# formula refers to RSH 5.3.1
# *** use StringIndexer to match the userId and animeId with the array index
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import avg
# from pyspark import StorageLevel

USERIndexer = StringIndexer(inputCol='userId', outputCol="userMap", handleInvalid="error",
                            stringOrderType="frequencyDesc")

ANIMEIndexer = StringIndexer(inputCol='animeId', outputCol="animeMap", handleInvalid="error",
                            stringOrderType="frequencyDesc")

USERModel = USERIndexer.fit(ratings)
user_map = USERModel.transform(ratings)\
                    .select('userMap', 'animeId', 'rating')\
                    .withColumnRenamed('userMap', 'userId')
user_set = user_map.withColumn('userId', user_map.userId.cast(IntegerType()))
# userId is settled

ANIMEModel = ANIMEIndexer.fit(user_set)
anime_map = ANIMEModel.transform(user_set)\
                      .select('userId', 'animeMap', 'rating')\
                      .withColumnRenamed('animeMap', 'animeId')
anime_set = anime_map.withColumn('animeId', anime_map.animeId.cast(IntegerType())).cache()
# animeId is settled

u_row = anime_set.select(avg('rating'))\
                 .withColumnRenamed('avg(rating)', 'mean')\
                 .collect()
u = u_row[0].mean
# Overall average rating

import numpy as np
Data = anime_set.repartition(10).rdd.cache() # TODO: find a better modification
user_num = anime_set.select('userId').distinct().count()
anime_num = anime_set.select('animeId').distinct().count()
user_bias_array = np.random.rand(user_num)
anime_bias_array = np.random.rand(anime_num)
user_bias = sc.broadcast(user_bias_array)
anime_bias = sc.broadcast(anime_bias_array)
# initialization of the user and item bias


# *** parallel sgd
lrate = 0.005
rvalue = 0.02
max_iteration = 3000
threshold = 0.0001
'''
narrow operation seems to follow no specific sequence when going through partitions
'''
def p_sgd(iterator):
    global u, lrate, rvalue, max_iteration, threshold
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    partition = []
    PartitionSize = 0
    for row in iterator:
        regis = [row.userId, row.animeId, row.rating]
        partition.append(regis)
        PartitionSize += 1
    # ****** iterator can only be executed once
    itr_cnt = 0
    fmr_rmse = 100
    cur_rmse = 0
    bu = user_bias.value
    bi = anime_bias.value
    while itr_cnt <= max_iteration and abs(cur_rmse-fmr_rmse) >= threshold:
        itr_cnt += 1
        fmr_rmse = cur_rmse
        cur_rmse = 0
        # shuffle
        np.random.shuffle(bu)
        np.random.shuffle(bi)
        # main logic
        for row in partition:
            predict_rating = u + bu[row[0]] + bi[row[1]]
            eui = row[2] - predict_rating
            cur_rmse += eui**2
            # gradient descent
            bu[row[0]] += lrate*(eui - rvalue*bu[row[0]])
            bi[row[1]] += lrate*(eui - rvalue*bi[row[1]])
        # calculate rmse
        cur_rmse = (cur_rmse / PartitionSize)**0.5
        print("current rmse: ", cur_rmse)
    yield bu, bi

rst = Data.mapPartitions(p_sgd).glom().collect()
# p_num = Data.rdd.getNumPartitions()

for index, bias in enumerate(rst):
    if index == 0:
        bu_rst = bias[0][0]
        bi_rst = bias[0][1]
    else:
        bu_rst += bias[0][0]
        bi_rst += bias[0][1]
bu_rst = bu_rst / 10
bi_rst = bi_rst / 10









