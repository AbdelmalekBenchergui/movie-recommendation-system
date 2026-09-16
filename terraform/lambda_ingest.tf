data "archive_file" "rate_ingest_lambda" {
  type        = "zip"
  source_file = "${path.module}/../scripts/rate_ingest.py"
  output_path = "${path.module}/../.build/rate_ingest_lambda.zip"
}

resource "aws_lambda_function" "rate_ingest" {
  function_name    = "${var.project}-${var.env}-rate-ingest"
  role             = aws_iam_role.lambda_role.arn
  runtime          = "python3.12"
  handler          = "rate_ingest.handler"
  timeout          = 30
  memory_size      = 128
  source_code_hash = data.archive_file.rate_ingest_lambda.output_base64sha256
  filename         = data.archive_file.rate_ingest_lambda.output_path

  environment {
    variables = {
      ML_BUCKET        = aws_s3_bucket.data_lake.bucket
      STREAMING_PREFIX = "raw/streaming/ratings"
    }
  }
}

resource "aws_apigatewayv2_integration" "rate" {
  api_id                 = aws_apigatewayv2_api.ml.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.rate_ingest.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "rate" {
  api_id    = aws_apigatewayv2_api.ml.id
  route_key = "POST /rate"
  target    = "integrations/${aws_apigatewayv2_integration.rate.id}"
}

resource "aws_lambda_permission" "rate_api" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.rate_ingest.function_name
  principal     = "apigateway.amazonaws.com"
  statement_id  = "rate-ingest-api-gw"
  source_arn    = "${aws_apigatewayv2_api.ml.execution_arn}/*"
}