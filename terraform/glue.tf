locals {
  glue_db_name = "${var.project}_${var.env}_db"
  catalog_id   = data.aws_caller_identity.current.account_id
}

data "aws_caller_identity" "current" {}

resource "aws_glue_catalog_database" "data_lake" {
  name        = local.glue_db_name
  description = "Data lake catalog for MovieLens recommendation system"
}

resource "aws_glue_catalog_table" "raw_ratings" {
  name          = "raw_ratings"
  database_name = aws_glue_catalog_database.data_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "csv"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data_lake.bucket}/raw/ratings/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
      parameters = {
        "field.delim"          = ";"
        "serialization.format" = ";"
      }
    }
    columns {
      name = "user_id"
      type = "bigint"
    }
    columns {
      name = "movie_id"
      type = "bigint"
    }
    columns {
      name = "rating"
      type = "double"
    }
    columns {
      name = "timestamp"
      type = "bigint"
    }
  }
}

resource "aws_glue_catalog_table" "raw_movies" {
  name          = "raw_movies"
  database_name = aws_glue_catalog_database.data_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "csv"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data_lake.bucket}/raw/movies/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
      parameters = {
        "field.delim"          = ";"
        "serialization.format" = ";"
      }
    }
    columns {
      name = "movie_id"
      type = "bigint"
    }
    columns {
      name = "title"
      type = "string"
    }
    columns {
      name = "genres"
      type = "string"
    }
  }
}

resource "aws_glue_catalog_table" "raw_users" {
  name          = "raw_users"
  database_name = aws_glue_catalog_database.data_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "csv"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data_lake.bucket}/raw/users/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
      parameters = {
        "field.delim"          = ";"
        "serialization.format" = ";"
      }
    }
    columns {
      name = "user_id"
      type = "bigint"
    }
    columns {
      name = "gender"
      type = "string"
    }
    columns {
      name = "age"
      type = "int"
    }
    columns {
      name = "occupation"
      type = "int"
    }
    columns {
      name = "zip_code"
      type = "string"
    }
  }
}

resource "aws_glue_catalog_table" "processed_ratings_fact" {
  name          = "ratings_fact"
  database_name = aws_glue_catalog_database.data_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "parquet"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data_lake.bucket}/processed/ratings_fact/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    columns {
      name = "user_id"
      type = "bigint"
    }
    columns {
      name = "movie_id"
      type = "bigint"
    }
    columns {
      name = "rating"
      type = "double"
    }
    columns {
      name = "timestamp"
      type = "bigint"
    }
    columns {
      name = "title"
      type = "string"
    }
    columns {
      name = "genres"
      type = "string"
    }
    columns {
      name = "gender"
      type = "string"
    }
    columns {
      name = "age"
      type = "int"
    }
    columns {
      name = "occupation"
      type = "int"
    }
  }

  partition_keys {
    name = "rating_date"
    type = "date"
  }
}

resource "aws_glue_catalog_table" "processed_movies" {
  name          = "movies"
  database_name = aws_glue_catalog_database.data_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "parquet"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data_lake.bucket}/processed/movies/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    columns {
      name = "movie_id"
      type = "bigint"
    }
    columns {
      name = "title"
      type = "string"
    }
    columns {
      name = "genres"
      type = "string"
    }
    columns {
      name = "genre_list"
      type = "array<string>"
    }
  }
}

resource "aws_glue_catalog_table" "processed_users" {
  name          = "users"
  database_name = aws_glue_catalog_database.data_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "parquet"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data_lake.bucket}/processed/users/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    columns {
      name = "user_id"
      type = "bigint"
    }
    columns {
      name = "gender"
      type = "string"
    }
    columns {
      name = "age"
      type = "int"
    }
    columns {
      name = "occupation"
      type = "int"
    }
    columns {
      name = "zip_code"
      type = "string"
    }
  }
}

resource "aws_glue_catalog_table" "processed_rating_features" {
  name          = "rating_features"
  database_name = aws_glue_catalog_database.data_lake.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    EXTERNAL       = "TRUE"
    classification = "parquet"
  }

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.data_lake.bucket}/processed/rating_features/"
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
    }
    columns {
      name = "user_id"
      type = "bigint"
    }
    columns {
      name = "ratings_count"
      type = "bigint"
    }
    columns {
      name = "avg_rating"
      type = "double"
    }
    columns {
      name = "first_rating_date"
      type = "date"
    }
    columns {
      name = "last_rating_date"
      type = "date"
    }
  }
}

resource "aws_s3_object" "glue_etl_script" {
  bucket = aws_s3_bucket.data_lake.bucket
  key    = "scripts/glue_etl_job.py"
  source = "${path.module}/../scripts/glue_etl_job.py"
  etag   = filemd5("${path.module}/../scripts/glue_etl_job.py")
}

resource "aws_glue_job" "etl" {
  name     = "${var.project}-${var.env}-etl"
  role_arn = aws_iam_role.glue_service_role.arn

  command {
    script_location = "s3://${aws_s3_bucket.data_lake.bucket}/scripts/glue_etl_job.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-bookmark-option"     = "job-bookmark-enable"
    "--output_path"             = "s3://${aws_s3_bucket.data_lake.bucket}/processed/"
    "--database"                = aws_glue_catalog_database.data_lake.name
    "--enable-glue-datacatalog" = "true"
    "--spark-event-logs-path"   = "s3://${aws_s3_bucket.data_lake.bucket}/spark-logs/"
  }

  max_retries       = 1
  timeout           = 60
  worker_type       = "G.1X"
  number_of_workers = 2
  glue_version      = "4.0"
}
