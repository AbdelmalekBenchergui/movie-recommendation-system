locals {
  sm_training_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-cpu-py311-ubuntu22.04-sagemaker-v1.11"
}

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = ["575267658029"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${var.project}-${var.env}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

data "aws_iam_policy_document" "sagemaker_execution" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:ListBucket",
      "s3:PutObject",
      "s3:DeleteObject"
    ]
    resources = [
      local.bucket_arn,
      "${local.bucket_arn}/*"
    ]
  }
  statement {
    actions = [
      "iam:GetRole",
      "iam:PassRole",
    ]
    resources = [aws_iam_role.sagemaker_execution.arn]
  }
  statement {
    actions = [
      "ecr:GetAuthorizationToken"
    ]
    resources = ["*"]
  }
  statement {
    actions = [
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchCheckLayerAvailability"
    ]
    resources = ["arn:aws:ecr:${var.aws_region}:763104351884:repository/pytorch-training"]
  }
  statement {
    actions = [
      "cloudwatch:PutMetricData"
    ]
    resources = ["*"]
  }
  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = [
      "arn:aws:logs:${var.aws_region}:*:log-group:/aws/sagemaker/*",
      "arn:aws:logs:${var.aws_region}:*:log-group:/aws/sagemaker/*:log-stream:*"
    ]
  }
}

resource "aws_iam_policy" "sagemaker_execution" {
  name        = "${var.project}-${var.env}-sagemaker-execution"
  description = "Allow SageMaker training to read input, write output, and pull the framework image"
  policy      = data.aws_iam_policy_document.sagemaker_execution.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution" {
  role       = aws_iam_role.sagemaker_execution.name
  policy_arn = aws_iam_policy.sagemaker_execution.arn
}