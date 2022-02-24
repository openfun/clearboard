
# Lambda invocation role
#########################

resource "aws_iam_role" "lambda_invocation_role" {
  name = "${terraform.workspace}-clearboard-lambda-invocation-role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Effect": "Allow"
    }
  ]
}
EOF
}

# Allow lambda to access ECR image
data "aws_ecr_repository" "clearboard_lambda_ecr" {
  name = "clearboard-lambda"
}

resource "aws_iam_policy" "lambda_ecr_access_policy" {
  name        = "${terraform.workspace}-clearboard-lambda-ecr-access-policy"
  path        = "/"
  description = "IAM policy needed by all lambda to access ECR"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "ecr:SetRepositoryPolicy",
        "ecr:GetRepositoryPolicy"
      ],
      "Effect": "Allow",
      "Resource": "${data.aws_ecr_repository.clearboard_lambda_ecr.arn}/"
    }
  ]
}
EOF
}


resource "aws_iam_role_policy_attachment" "lambda_invocation_ecr_policy_attachment" {
  role       = aws_iam_role.lambda_invocation_role.name
  policy_arn = aws_iam_policy.lambda_ecr_access_policy.arn
}


resource "aws_iam_policy" "lambda_logging_policy" {
  name        = "${terraform.workspace}-clearboard-lambda-logging-policy"
  path        = "/"
  description = "IAM policy for logging from a lambda"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*",
      "Effect": "Allow"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "lambda_logging_policy_attachment" {
  role       = aws_iam_role.lambda_invocation_role.name
  policy_arn = aws_iam_policy.lambda_logging_policy.arn
}

resource "aws_iam_policy" "lambda_s3_access_policy" {
  name        = "${terraform.workspace}-clearboard-lambda-s3-access-policy"
  path        = "/"
  description = "IAM policy to read in source bucket and write in destination bucket"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": ["s3:GetObject"],
      "Effect": "Allow",
      "Resource": "arn:aws:s3:::${aws_s3_bucket.clearboard_source.bucket}/*"
    },
    {
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Effect": "Allow",
      "Resource": "arn:aws:s3:::${aws_s3_bucket.clearboard_destination.bucket}/*"
    }
  ]
}
EOF
}

# `lambda-enhance` needs read access to the source bucket and write access to the
# destination bucket
resource "aws_iam_role_policy_attachment" "lambda_s3_access_policy_attachment" {
  role        = aws_iam_role.lambda_invocation_role.name
  policy_arn  = aws_iam_policy.lambda_s3_access_policy.arn
}


# Event rule role
###################

resource "aws_iam_role" "event_rule_role" {
  name = "${terraform.workspace}-clearboard-event-rule-role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "events.amazonaws.com"
      },
      "Effect": "Allow"
    }
  ]
}
EOF
}

resource "aws_iam_policy" "event_rule_lambda_invoke_policy" {
  name        = "${terraform.workspace}-clearboard-event-lambda-invoke-policy"
  path        = "/"
  description = "IAM policy for invoking a lambda from an event rule"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "lambda:InvokeFunction"
      ],
      "Resource": "${aws_lambda_function.clearboard_enhance_lambda.arn}",
      "Effect": "Allow"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "event_rule_lambda_invoke_policy_attachment" {
  role       = aws_iam_role.event_rule_role.name
  policy_arn = aws_iam_policy.event_rule_lambda_invoke_policy.arn
}

