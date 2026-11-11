import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, to_date, year, month, date_format,
    monotonically_increasing_id, round as spark_round,
    broadcast, row_number
)
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

# 1. Initialize PySpark Session
spark = SparkSession.builder \
    .appName("RealtorLakehousePipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(">>> [1/4] Spark Session Initialized.")

# 2. Extract Raw Dataset (uses local full CSV if available, else 50-row sample)
full_csv = "realtor-data.zip.csv"
onedrive_csv = r"C:\Users\Ganesh Sharma\OneDrive\Documents\Data Engineering Project\realtor-data.zip.csv"
sample_csv = os.path.join("sample_data", "realtor_sample_50.csv")

if os.path.exists(full_csv):
    csv_path = full_csv
elif os.path.exists(onedrive_csv):
    csv_path = onedrive_csv
else:
    csv_path = sample_csv

df_raw = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(csv_path)

raw_count = df_raw.count()
print(f">>> [2/4] Raw records ingested: {raw_count:,} rows")

# 3. Transform: Data Cleaning, Filtering & Window Deduplication
valid_rows = (
    col("price").isNotNull() & (col("price") > 0) &
    col("house_size").isNotNull() & (col("house_size") > 0) &
    col("state").isNotNull() & col("city").isNotNull()
)

df_filtered = df_raw.filter(valid_rows)
dropped_count = raw_count - df_filtered.count()

# Remove duplicate property listings, keeping the latest record by sold date
dedup_spec = Window.partitionBy("street", "city", "state", "zip_code", "price") \
                   .orderBy(col("prev_sold_date").desc_nulls_last())

df_clean = df_filtered \
    .withColumn("rn", row_number().over(dedup_spec)) \
    .filter(col("rn") == 1) \
    .drop("rn") \
    .withColumn("bed", when(col("bed").isNull(), 0).otherwise(col("bed").cast(IntegerType()))) \
    .withColumn("bath", when(col("bath").isNull(), 0).otherwise(col("bath").cast(IntegerType()))) \
    .withColumn("price_per_sqft", spark_round(col("price") / col("house_size"), 2)) \
    .withColumn("sold_date", to_date(col("prev_sold_date"), "yyyy-MM-dd"))

print(f">>> [3/4] Cleaned dataset: {df_clean.count():,} unique rows ({dropped_count:,} incomplete rows filtered)")

# 4. Data Modeling: Build Star Schema (Dimensions & Fact Table)
dim_location = df_clean.select("city", "state", "zip_code").distinct() \
    .withColumn("location_id", monotonically_increasing_id())

dim_property = df_clean.select("status", "bed", "bath", "acre_lot", "house_size").distinct() \
    .withColumn("property_id", monotonically_increasing_id())

dim_date = df_clean.select("sold_date").where(col("sold_date").isNotNull()).distinct() \
    .withColumn("date_id", monotonically_increasing_id()) \
    .withColumn("year", year(col("sold_date"))) \
    .withColumn("month", month(col("sold_date"))) \
    .withColumn("quarter", date_format(col("sold_date"), "Q"))

fact_listings = df_clean \
    .join(broadcast(dim_location), ["city", "state", "zip_code"], "inner") \
    .join(broadcast(dim_property), ["status", "bed", "bath", "acre_lot", "house_size"], "inner") \
    .join(broadcast(dim_date), ["sold_date"], "left") \
    .select(
        monotonically_increasing_id().alias("listing_id"),
        "location_id",
        "property_id",
        "date_id",
        "state",
        "price",
        "house_size",
        "price_per_sqft"
    )

# 5. Spark SQL Analytical Query
fact_listings.createOrReplaceTempView("fact_listings")

query = """
    SELECT 
        state,
        COUNT(listing_id) AS total_listings,
        ROUND(AVG(price), 2) AS avg_listing_price,
        ROUND(AVG(price_per_sqft), 2) AS avg_price_per_sqft,
        DENSE_RANK() OVER (ORDER BY AVG(price) DESC) AS state_price_rank
    FROM fact_listings
    GROUP BY state
    HAVING COUNT(listing_id) > 100
    ORDER BY state_price_rank ASC
    LIMIT 10
"""

print(">>> [4/4] Running Spark SQL state valuation ranking:")
spark.sql(query).show(truncate=False)

print(">>> Pipeline completed successfully!")
spark.stop()