resource "aws_iam_policy" "rate_ingest_s3" {
  name        = "${var.project}-${var.env}-rate-ingest-s3"
  description = "Allow the rate ingest Lambda to write ratings to the lake streaming zone"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "s3:PutObject"
      Resource = "${aws_s3_bucket.data_lake.arn}/raw/streaming/ratings/*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "rate_ingest_s3" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.rate_ingest_s3.arn
}