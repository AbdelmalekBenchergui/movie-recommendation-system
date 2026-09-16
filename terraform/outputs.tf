output "bucket_name" {
  description = "Data lake S3 bucket"
  value       = aws_s3_bucket.data_lake.id
}

output "bucket_raw_prefix" {
  description = "Raw zone URL"
  value       = "s3://${aws_s3_bucket.data_lake.id}/raw/"
}

output "bucket_processed_prefix" {
  description = "Processed zone URL"
  value       = "s3://${aws_s3_bucket.data_lake.id}/processed/"
}

output "glue_database" {
  description = "Glue Catalog database name"
  value       = aws_glue_catalog_database.data_lake.name
}

output "glue_job_name" {
  description = "Glue ETL job name"
  value       = aws_glue_job.etl.name
}

output "athena_workgroup" {
  description = "Athena workgroup name"
  value       = aws_athena_workgroup.data_lake.name
}

output "glue_role_arn" {
  description = "Glue service role ARN"
  value       = aws_iam_role.glue_service_role.arn
}

output "sagemaker_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution.arn
}

output "api_url" {
  description = "Recommendations API endpoint"
  value       = "${aws_apigatewayv2_api.ml.api_endpoint}/prod"
}
