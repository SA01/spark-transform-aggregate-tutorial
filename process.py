from pyspark.sql import DataFrame
import pyspark.sql.functions as sql_func
from pyspark.sql.types import IntegerType, ArrayType, DateType, StringType, StructType, StructField
from pyspark import SparkConf
from pyspark.sql import SparkSession


def create_spark_session() -> SparkSession:
    conf = SparkConf().set("spark.driver.memory", "8g")

    spark_session = SparkSession\
        .builder\
        .master("local[4]")\
        .config(conf=conf)\
        .appName("Aggregate Transform Tutorial") \
        .getOrCreate()

    return spark_session


def double_numbers(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "double",
        sql_func.transform(sql_func.col("numbers"), (lambda num: sql_func.cast(IntegerType(), num) * 2))
    )


def days_since_first_event(df: DataFrame) -> DataFrame:
    return df\
        .withColumn("first_event", sql_func.array_min("dates"))\
        .withColumn(
            "days_since_first_event",
            sql_func.transform(sql_func.col("dates"), lambda dt: sql_func.datediff(dt, sql_func.col("first_event")))
        )


def sum_numbers(df: DataFrame) -> DataFrame:
    return df.withColumn("sum", sql_func.aggregate(sql_func.col("numbers"),
                                                   initialValue=sql_func.lit(0),
                                                   merge=lambda acc, val: acc + val)
                         )


def sum_numbers_init_val(df: DataFrame) -> DataFrame:
    int_df = df\
        .withColumn("start", sql_func.monotonically_increasing_id())

    int_df.show(truncate=False)

    return int_df\
        .withColumn(
            "sum",
            sql_func.aggregate(
                sql_func.col("numbers"),
                initialValue=sql_func.col("start"),
                merge=lambda acc, val: acc + val,
                finish=lambda val: val * 2
            )
        )


def create_latest_state(df: DataFrame) -> DataFrame:
    return df\
        .withColumn(
            "ownership",
            sql_func.expr(
                "array_sort(ownership, (left, right) -> "
                "case when left.transfer_date < right.transfer_date then -1 "
                "when left.transfer_date > right.transfer_date then 1 "
                "else 0 end)"
                )
        )\
        .withColumn("first_ownership", sql_func.element_at("ownership", 1))\
        .withColumn("current_ownership", sql_func.element_at("ownership", -1))\
        .withColumn("first_state", sql_func.col("first_ownership.state"))\
        .withColumn(
            "current_state",
            sql_func.aggregate(
                sql_func.col("ownership"),
                sql_func.col("first_state"),
                lambda acc, event: sql_func.when(event['state'].isNotNull(), event['state']).otherwise(acc)
            )
        )\
        .withColumn(
            "current_ownership",
            sql_func.col("current_ownership").withField("state", sql_func.col("current_state"))
        )\
        .drop("first_state", "current_state")


if __name__ == '__main__':
    spark = create_spark_session()

    # transform on numbers example
    numbers_data = spark.read.option("header", "true").csv("data/numbers.csv")\
        .withColumn("numbers", sql_func.from_json(sql_func.col("numbers"), schema=ArrayType(IntegerType())))
    numbers_data.printSchema()
    numbers_data.show(truncate=False)

    doubled = numbers_data.transform(double_numbers)
    doubled.show(truncate=False)

    # Days since first event using transform example
    dates_data = spark.read.option("header", "true").csv("data/dates.csv") \
        .withColumn("dates", sql_func.from_json(sql_func.col("dates"), schema=ArrayType(DateType())))
    dates_data.printSchema()
    dates_data.show(truncate=False)

    days_since_event = dates_data.transform(days_since_first_event)
    days_since_event.show(truncate=False)

    # Sum using aggregate example
    sum_numbers_df = numbers_data.transform(sum_numbers_init_val)
    sum_numbers_df.show(truncate=False)

    # Car ownership example

    # This schema will be used to convert json array as string to struct
    data_schema = ArrayType(
        StructType(
            [
                StructField("owner_name", StringType()),
                StructField("transfer_date", DateType()),
                StructField("state", StringType(), nullable=True)
            ]
        )
    )

    car_reg_data = spark.read.option("header", "true").csv("data/cars_registration.csv") \
        .withColumn("date_of_manufacture", sql_func.to_date(sql_func.col("date_of_manufacture"))) \
        .withColumn("ownership", sql_func.from_json(sql_func.col("ownership"), schema=data_schema))

    car_reg_data.show(truncate=False)
    car_reg_data.printSchema()

    processed_reg = car_reg_data.transform(create_latest_state)
    processed_reg.show(truncate=False)
