from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType

RATIING_DATASET = "animelists_cleaned.csv"
USER_DATASET = "users_cleaned.csv"

UserAnimeSchema = StructType([
    StructField("username", StringType(), True),
    StructField("animeId", IntegerType(), True),
    StructField("watched_episodes", IntegerType(), True),
    StructField("start_date", StringType(), True),
    StructField("finish_date", StringType(), True),
    StructField("rating", IntegerType(), True),
    StructField("status", IntegerType(), True),
    StructField("rewatching", StringType(), True),
    StructField("rewatching_ep", LongType(), True),
    StructField("last_updated_date", StringType(), True),
    StructField("tags", StringType(), True),
])

user_anime = spark.read.schema(UserAnimeSchema)\
                  .option("header", "false")\
                  .option("mode", "DROPMALFORMED")\
                  .csv(RATIING_DATASET)

user_anime_cleaned = user_anime.select("username", "animeId", "watched_episodes").na.drop()

users = spark.read.csv(USER_DATASET,header = "true", inferSchema="true")\
             .select("username", "user_id")\
             .na.drop()

ratings = user_anime_cleaned.join(users, "username")\
                            .select("user_id", "animeId", "rating")\
                            .withColumnRenamed("user_id", "userId")
# three columns "userId", "animeId", "rating" all in integers and nullable
# pure ratings without biases

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

#miniset = ratings.randomSplit([0.5, 0.5])[0]
(training, test) = ratings.randomSplit([0.7, 0.3])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(rank = 10, maxIter = 5, regParam = 0.01,
          numUserBlocks = 10, numItemBlocks = 10,
          userCol = "userId", itemCol = "animeId",
          ratingCol = "rating",
          coldStartStrategy = "drop")
rsmodel = als.fit(training)
#rsmodel_path = "/Users/soober/Documents/HKUST_MSc_bdt/2019_Spring/MSBD5003_Big_Data_Computing/Project"
#rsmodel.save(rsmodel_path)


# Evaluate the model by computing the RMSE on the test data
predictions = rsmodel.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

userRecs = rsmodel.recommendForAllUsers(10)
userRecs.collect()


# Retrieve the result and formalize the prediction
# TODO: choose a nice comparison between SGD and ALS on SVD++
a = userRecs.collect()
i = 1
for item in a:
    j = 1
    rst = []
    for anime in item.recommendations:
        rst.append([anime.animeId, anime.rating])
        j += 1
        if j == 5:
            break
    print("(", item.userId, ",", rst, ")")
    i += 1
    if i == 6:
        break

rowlist = userRecs.collect()
rst = []
for row in rowlist:
    temp = [row.userId]
    for a in row.recommendations:
        temp.append(a.animeId)
        temp.append(a.rating)
    rst.append(temp)
df = spark.createDataFrame(rst, ["userId", "a1", "a1_rating",
                                 "a2", "a2_rating",
                                 "a3", "a3_rating",
                                 "a4", "a4_rating",
                                 "a5", "a5_rating",
                                 "a6", "a6_rating",
                                 "a7", "a7_rating",
                                 "a8", "a8_rating",
                                 "a9", "a9_rating",
                                 "a10", "a10_rating"])