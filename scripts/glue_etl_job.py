#!/usr/bin/env python3
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "output_path", "database"],
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

DB = args["database"]
OUT = args["output_path"].rstrip("/")

df_ratings = (
    glueContext.create_dynamic_frame
    .from_catalog(database=DB, table_name="raw_ratings")
    .toDF()
)

df_movies = (
    glueContext.create_dynamic_frame
    .from_catalog(database=DB, table_name="raw_movies")
    .toDF()
)

df_users = (
    glueContext.create_dynamic_frame
    .from_catalog(database=DB, table_name="raw_users")
    .toDF()
)

df_ratings = (
    df_ratings
    .dropDuplicates(["user_id", "movie_id"])
    .filter(F.col("rating").isNotNull())
    .withColumn("rating_date", F.to_timestamp(F.col("timestamp")).cast("date"))
)

df_movies = df_movies.dropDuplicates(["movie_id"]).filter(F.col("title").isNotNull())

df_movies = df_movies.withColumn(
    "genre_list", F.split(F.col("genres"), "\\|")
)

rating_features = (
    df_ratings
    .groupBy("user_id")
    .agg(
        F.count("rating").alias("ratings_count"),
        F.avg("rating").alias("avg_rating"),
        F.min("rating_date").alias("first_rating_date"),
        F.max("rating_date").alias("last_rating_date"),
    )
)

df_fact = (
    df_ratings
    .join(df_movies.select("movie_id", "title", "genres"), on="movie_id", how="left")
    .join(df_users.select("user_id", "gender", "age", "occupation"), on="user_id", how="left")
    .select(
        "user_id", "movie_id", "rating", "timestamp", "rating_date",
        "title", "genres", "gender", "age", "occupation"
    )
)

df_fact.write.mode("overwrite").partitionBy("rating_date").parquet(f"{OUT}/ratings_fact")

df_movies.write.mode("overwrite").parquet(f"{OUT}/movies")
df_users.write.mode("overwrite").parquet(f"{OUT}/users")
rating_features.write.mode("overwrite").parquet(f"{OUT}/rating_features")

job.commit()
print(f"ETL complete. Output written to {OUT}")
