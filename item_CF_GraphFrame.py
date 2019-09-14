# Item_based Collaborative Filtering ---GraphFrames, DataFrame
sc.addPyFile("/Users/soober/Downloads/graphframes-0.7.0-spark2.4-s_2.11.jar")

from graphframes import *
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType

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

user_anime = spark.read.schema(UserAnimeSchema) \
    .option("header", "false") \
    .option("mode", "DROPMALFORMED") \
    .csv(RATIING_DATASET)

raw_edges = user_anime.select("username", "animeId", "rating") \
    .na.drop(subset=["username", "animeId"]) \
    .filter("rating <= 10 and rating >= 1")

userlist = spark.read.csv(USER_DATASET, header="true", inferSchema="true") \
    .select("username", "user_id") \
    .na.drop()

edges = raw_edges.join(userlist, "username") \
    .select("user_id", "animeId", "rating") \
    .withColumnRenamed("user_id", "src") \
    .withColumnRenamed("animeId", 'dst')

# ......................Plan A to get smaller dataset...................
# split_factor = 0.001
# miniset = edges.randomSplit([split_factor, 1 - split_factor])[0]
# print("Current graph contains ", miniset.count(), " edges.")

# user = miniset.select("src").distinct().withColumnRenamed("src", "id")
# anime = miniset.select("dst").distinct().withColumnRenamed("dst", "id")
# print("Current graph contains ", user.count(), " users.")
# print("Current graph contains ", anime.count(), " animes.")
# vertices = user.join(anime, "id", 'outer')
# g = GraphFrame(vertices, miniset)
# ......................................................................


# STATS:
# 19172125 edges
# 106402 distinct users
# 6598 distinct animes
# should be 113000 vertices in total
# 111958 in fact  1042 overlapped
# overlapped ids will not matter here since the structure of the graph is ignored


# animelist = spark.read\
#                 .csv("/Users/soober/Documents/HKUST_MSc_bdt/2019_Spring/MSBD5003_Big_Data_Computing/MyAnimeList Dataset/anime_cleaned.csv",
#                     header = "true", inferSchema="true")\
#                 .select("anime_id")\
#                 .na.drop(subset = ["anime_id"])\
#                 .withColumnRenamed("anime_id", "id")


# the graph is actually a bipartite graph
# no bipartite graph implemented in graphframes so compromised by using outer join

from pyspark.sql.functions import lit

# still need to limit nodes specifically other than random split edges
user_num = 20000
anime_num = 1000
uc = edges.select("src").distinct().head(user_num)
user = spark.createDataFrame(uc)
ac = edges.select("dst").distinct().head(anime_num)
anime = spark.createDataFrame(ac)

user_to_join = user.withColumn("aux1", lit(1)).withColumn("aux2", lit(1))
anime_to_join = anime.withColumn("aux1", lit(1)).withColumn("aux2", lit(1))

new_edges = edges.select("src", "dst", "rating") \
    .join(user_to_join, "src") \
    .join(anime_to_join, "dst") \
    .select("src", "dst", "rating")
# 454 animes in total

user_v = user.withColumnRenamed("src", "id")
anime_v = anime.withColumnRenamed("dst", "id")
vertices = user_v.join(anime_v, "id", 'outer')
g = GraphFrame(vertices, new_edges)

from pyspark.sql.functions import array
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, ArrayType, DoubleType, StringType
import numpy as np
import pandas as pd

simSchema = StructType([
    StructField("anime_pair", ArrayType(IntegerType()), True),
    StructField("simVal", DoubleType(), True)])


@pandas_udf(simSchema, PandasUDFType.GROUPED_MAP)
def calsim(pdf):
    pair = pdf.rating_pair
    sum_xx, sum_xy, sum_yy = (0.0, 0.0, 0.0)
    for rating_pair in pair:
        sum_xx += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_yy += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])
    denominator = np.sqrt(sum_xx) * np.sqrt(sum_yy)
    return pd.DataFrame([[pdf.anime_pair[0], (sum_xy / (float(denominator))) if denominator else 0.0]])


# calculate the similarity between two animes
# if two rating columns can be transformed into dense vector the cosine similarity is just dot(v1, v2)/v1.norm*v2.norm

anime_rating_pair = g.find("(a)-[r1]->(b); (a)-[r2]->(c)") \
    .where('b != c') \
    .select(array("r1.rating", "r2.rating").alias("rating_pair"),
            array("b.id", "c.id").alias("anime_pair"))
#                      .persist(storageLevel=StorageLevel(False, True, False, False, 1))

anime_sim = anime_rating_pair.groupBy("anime_pair").apply(calsim)

# need to cut the redundancy otherwise the local storage is not able to hold
# (the system storage was almost consumed ~200G, too large shuffle write into the memory)
# anime_sim: anime_pair([int, int]) -> simVal(double)

from collections import defaultdict

selectSchema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("recs", ArrayType(ArrayType(IntegerType())), True)])

threshold = 50  # no more than x recommendations will be given to each user


@pandas_udf(selectSchema, PandasUDFType.GROUPED_MAP)
def caltop(pdf):
    # first aggregate on c.id
    numerator = defaultdict(int)
    denominator = defaultdict(int)
    # then calculate the sum of rating*sim and sim separately
    
    for triple in pdf.reqinfo:
        numerator[triple[2]] += triple[0] * triple[1]
        denominator[triple[2]] += triple[1]
    
    recs = [(anime, num / denominator[anime]) for anime, num in numerator.items()]
    recs.sort(key=lambda x: x[1], reverse=True)
    if len(recs) > threshold:
        recs = recs[:threshold]
    
    return pd.DataFrame([[pdf.id[0], recs]])
    # return the userid and its associate list of recommended animes which might have already been watched


sim_edges = anime_sim.withColumn("src", anime_sim.anime_pair[0]) \
    .withColumn("dst", anime_sim.anime_pair[1]) \
    .select("src", "dst", "simVal")

increment_edges = new_edges.join(sim_edges, ["src", "dst"], 'outer')

ng = GraphFrame(vertices, increment_edges)
# didn't find operations like add-edges on a given graph


from pyspark.sql.functions import array, lit, concat_ws, split

rec_prepare = ng.find("(a)-[r]->(b); (b)-[s]->(c)") \
    .where("r.rating > 0 AND s.simVal > 0") \
    .select("a.id", array("r.rating", "s.simVal", "c.id").alias("reqinfo"))
# rec_prepare.show()
user_rec = rec_prepare.groupBy("id").apply(caltop)
# user_rec should be userId ---> list of [anime, value]
# the value can be treated as the prediction of rating

# **********
# if one user has only watched one anime which is never watched by other users
# then there will not be recommendations for him/her
# **********

a = user_rec.collect()
i = 1
for item in a:
    print("(", item.userId, ",", item.recs[:5], ")")
    i += 1
    if i == 10:
        break
