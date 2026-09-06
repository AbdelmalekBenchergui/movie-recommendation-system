resource "aws_s3_bucket" "athena_results" {
  bucket        = "${var.project}-${var.env}-athena-results"
  force_destroy = var.env == "dev" ? true : false
}

resource "aws_athena_workgroup" "data_lake" {
  name = "${var.project}-${var.env}-athena"

  configuration {
    engine_version {
      selected_engine_version = "AUTO"
    }

    result_configuration {
      output_location = "s3://${aws_s3_bucket.athena_results.bucket}/results/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }
}
