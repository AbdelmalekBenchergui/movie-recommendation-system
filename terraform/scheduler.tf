resource "aws_iam_role" "eventbridge_sfn" {
  name = "${var.project}-${var.env}-eb-sfn-retrain"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "events.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_policy" "eventbridge_sfn" {
  name        = "${var.project}-${var.env}-eb-sfn-retrain"
  description = "Allow EventBridge to start the retrain state machine"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "states:StartExecution"
      Resource = aws_sfn_state_machine.retrain.arn
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eventbridge_sfn" {
  role       = aws_iam_role.eventbridge_sfn.name
  policy_arn = aws_iam_policy.eventbridge_sfn.arn
}

resource "aws_cloudwatch_event_rule" "weekly_retrain" {
  name                = "${var.project}-${var.env}-weekly-retrain"
  description         = "Kick off the weekly recommendation model retrain"
  schedule_expression = "cron(0 3 ? * MON *)"
}

resource "aws_cloudwatch_event_target" "weekly_retrain" {
  rule     = aws_cloudwatch_event_rule.weekly_retrain.name
  arn      = aws_sfn_state_machine.retrain.arn
  role_arn = aws_iam_role.eventbridge_sfn.arn
  input    = jsonencode({ instance_type = "ml.m5.large", epochs = "25" })
}