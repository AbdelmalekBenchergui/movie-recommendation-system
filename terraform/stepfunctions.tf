data "archive_file" "deploy_lambda" {
  type        = "zip"
  source_file = "${path.module}/../scripts/deploy_lambda.py"
  output_path = "${path.module}/../.build/deploy_lambda.zip"
}

data "archive_file" "verify_lambda" {
  type        = "zip"
  source_file = "${path.module}/../scripts/verify_lambda.py"
  output_path = "${path.module}/../.build/verify_lambda.zip"
}

data "archive_file" "prepare_lambda" {
  type        = "zip"
  source_file = "${path.module}/../scripts/prepare_data.py"
  output_path = "${path.module}/../.build/prepare_lambda.zip"
}

resource "aws_ssm_parameter" "model_prefix" {
  name  = "/${var.project}/${var.env}/model-prefix"
  type  = "String"
  value = var.model_prefix

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_iam_role" "sfn_role" {
  name = "${var.project}-${var.env}-sfn-retrain"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "states.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_policy" "sfn_retrain" {
  name        = "${var.project}-${var.env}-sfn-retrain"
  description = "Permit the retrain state machine to run SageMaker jobs, pass its role, and invoke deploy/verify Lambdas"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:AddTags",
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:StopProcessingJob",
          "glue:StartJobRun",
          "glue:GetJobRun",
          "glue:GetJob",
          "glue:BatchStopJobRun",
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = aws_iam_role.sagemaker_execution.arn
      },
      {
        Effect = "Allow"
        Action = "lambda:InvokeFunction"
        Resource = [
          aws_lambda_function.deploy.arn,
          aws_lambda_function.verify.arn,
          aws_lambda_function.prepare.arn,
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "events:PutRule",
          "events:PutTargets",
          "events:DeleteRule",
          "events:RemoveTargets",
          "events:DescribeRule",
          "events:TagResource",
          "events:UntagResource",
        ]
        Resource = "arn:aws:events:*:*:rule/StepFunctionsGetEventsFor*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sfn_retrain" {
  role       = aws_iam_role.sfn_role.name
  policy_arn = aws_iam_policy.sfn_retrain.arn
}

resource "aws_iam_role" "sfn_lambda_role" {
  name = "${var.project}-${var.env}-sfn-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sfn_lambda_basic" {
  role       = aws_iam_role.sfn_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_policy" "sfn_lambda_ops" {
  name = "${var.project}-${var.env}-sfn-lambda-ops"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "lambda:UpdateFunctionConfiguration"
        Resource = aws_lambda_function.recommend.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:PutParameter",
        ]
        Resource = aws_ssm_parameter.model_prefix.arn
      },
      {
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetPartitions",
          "glue:GetTable",
        ]
        Resource = [
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:catalog",
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:database/${aws_glue_catalog_database.data_lake.name}",
          "arn:aws:glue:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/${aws_glue_catalog_database.data_lake.name}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:AbortMultipartUpload",
        ]
        Resource = [
          aws_s3_bucket.athena_results.arn,
          "${aws_s3_bucket.athena_results.arn}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:ListBucket",
        ]
        Resource = aws_s3_bucket.data_lake.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:AbortMultipartUpload",
        ]
        Resource = "${aws_s3_bucket.data_lake.arn}/ml/input/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
        ]
        Resource = "${aws_s3_bucket.data_lake.arn}/processed/*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sfn_lambda_ops" {
  role       = aws_iam_role.sfn_lambda_role.name
  policy_arn = aws_iam_policy.sfn_lambda_ops.arn
}

resource "aws_lambda_function" "deploy" {
  function_name    = "${var.project}-${var.env}-retrain-deploy"
  role             = aws_iam_role.sfn_lambda_role.arn
  runtime          = "python3.12"
  handler          = "deploy_lambda.handler"
  timeout          = 60
  memory_size      = 128
  source_code_hash = data.archive_file.deploy_lambda.output_base64sha256
  filename         = data.archive_file.deploy_lambda.output_path

  environment {
    variables = {
      ML_BUCKET               = aws_s3_bucket.data_lake.bucket
      RECOMMEND_FUNCTION_NAME = aws_lambda_function.recommend.function_name
      MODEL_PREFIX_PARAM      = aws_ssm_parameter.model_prefix.name
    }
  }
}

resource "aws_lambda_function" "verify" {
  function_name    = "${var.project}-${var.env}-retrain-verify"
  role             = aws_iam_role.sfn_lambda_role.arn
  runtime          = "python3.12"
  handler          = "verify_lambda.handler"
  timeout          = 120
  memory_size      = 128
  source_code_hash = data.archive_file.verify_lambda.output_base64sha256
  filename         = data.archive_file.verify_lambda.output_path

  environment {
    variables = {
      API_URL = "${aws_apigatewayv2_api.ml.api_endpoint}/prod"
    }
  }
}

resource "aws_lambda_function" "prepare" {
  function_name    = "${var.project}-${var.env}-retrain-prepare"
  role             = aws_iam_role.sfn_lambda_role.arn
  runtime          = "python3.12"
  handler          = "prepare_data.handler"
  timeout          = 120
  memory_size      = 256
  source_code_hash = data.archive_file.prepare_lambda.output_base64sha256
  filename         = data.archive_file.prepare_lambda.output_path

  environment {
    variables = {
      ML_BUCKET        = aws_s3_bucket.data_lake.bucket
      GLUE_DATABASE    = aws_glue_catalog_database.data_lake.name
      ATHENA_WORKGROUP = aws_athena_workgroup.data_lake.name
      INPUT_S3_PREFIX  = "ml/input"
    }
  }
}

resource "aws_lambda_permission" "sfn_deploy" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.deploy.function_name
  principal     = "states.amazonaws.com"
  statement_id  = "sfn-retrain-deploy"
  source_arn    = aws_sfn_state_machine.retrain.arn
}

resource "aws_lambda_permission" "sfn_verify" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.verify.function_name
  principal     = "states.amazonaws.com"
  statement_id  = "sfn-retrain-verify"
  source_arn    = aws_sfn_state_machine.retrain.arn
}

resource "aws_lambda_permission" "sfn_prepare" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.prepare.function_name
  principal     = "states.amazonaws.com"
  statement_id  = "sfn-retrain-prepare"
  source_arn    = aws_sfn_state_machine.retrain.arn
}

resource "aws_sfn_state_machine" "retrain" {
  name     = "${var.project}-${var.env}-retrain"
  role_arn = aws_iam_role.sfn_role.arn
  definition = templatefile("${path.module}/state_machine.json", {
    sagemaker_role_arn = aws_iam_role.sagemaker_execution.arn
    training_image     = local.sm_training_image
    output_path        = "s3://${aws_s3_bucket.data_lake.id}/ml/models"
    bucket             = aws_s3_bucket.data_lake.id
    code_s3            = "s3://${aws_s3_bucket.data_lake.id}/ml/retrain-code"
    deploy_lambda_arn  = aws_lambda_function.deploy.arn
    verify_lambda_arn  = aws_lambda_function.verify.arn
    prepare_lambda_arn = aws_lambda_function.prepare.arn
    etl_job_name       = aws_glue_job.etl.name
  })
}