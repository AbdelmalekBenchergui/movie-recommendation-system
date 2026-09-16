data "archive_file" "recommend_lambda" {
  type        = "zip"
  source_dir  = "${path.module}/../.build/lambda_pkg"
  output_path = "${path.module}/../.build/recommend_lambda.zip"
}

resource "aws_lambda_function" "recommend" {
  function_name    = "${var.project}-${var.env}-recommend"
  role             = aws_iam_role.lambda_role.arn
  runtime          = "python3.12"
  handler          = "recommend.handler"
  timeout          = 30
  memory_size      = 512
  source_code_hash = data.archive_file.recommend_lambda.output_base64sha256
  filename         = data.archive_file.recommend_lambda.output_path

  environment {
    variables = {
      ML_BUCKET    = aws_s3_bucket.data_lake.bucket
      MODEL_PREFIX = var.model_prefix
    }
  }

  lifecycle {
    ignore_changes = [environment]
  }
}

resource "aws_apigatewayv2_api" "ml" {
  name          = "${var.project}-${var.env}-ml-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "recommend" {
  api_id                 = aws_apigatewayv2_api.ml.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.recommend.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "recommendations" {
  api_id    = aws_apigatewayv2_api.ml.id
  route_key = "GET /recommendations"
  target    = "integrations/${aws_apigatewayv2_integration.recommend.id}"
}

resource "aws_apigatewayv2_route" "similar" {
  api_id    = aws_apigatewayv2_api.ml.id
  route_key = "GET /similar"
  target    = "integrations/${aws_apigatewayv2_integration.recommend.id}"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.ml.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.recommend.id}"
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.ml.id
  name        = "prod"
  auto_deploy = true
}

resource "aws_lambda_permission" "recommend_api" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.recommend.function_name
  principal     = "apigateway.amazonaws.com"
  statement_id  = "recommend-api-gw"
  source_arn    = "${aws_apigatewayv2_api.ml.execution_arn}/*"
}