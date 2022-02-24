
# Enhancing
###########

resource "aws_lambda_function" "clearboard_enhance_lambda" {
  function_name    = "${terraform.workspace}-clearboard-enhance"
  image_uri        = "${data.aws_ecr_repository.clearboard_lambda_ecr.repository_url}:${terraform.workspace}"
  package_type     = "Image"
  role             = aws_iam_role.lambda_invocation_role.arn
  memory_size      = "1536"
  timeout          = "90"

  image_config {
    command = ["lambda-enhance.app.handler"]
  }

  environment {
    variables = {
      ENV_TYPE = terraform.workspace
      S3_DESTINATION_BUCKET   = aws_s3_bucket.clearboard_destination.id
    }
  }
}

resource "aws_lambda_permission" "allow_bucket" {
  statement_id  = "AllowExecutionFromS3Bucket"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.clearboard_enhance_lambda.arn
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.clearboard_source.arn
}
