# Create source S3 Bucket for uploaded images
resource "aws_s3_bucket" "clearboard_source" {
  bucket = "${terraform.workspace}-clearboard-source-${var.s3_bucket_unique_suffix}"
  acl    = "private"

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["POST"]
    allowed_origins = ["*"]
    max_age_seconds = 3600
  }

  tags = {
    Name        = "clearboard-source"
    Environment = terraform.workspace
  }
}

# Create destination S3 Bucket for enhanced images
resource "aws_s3_bucket" "clearboard_destination" {
  bucket = "${terraform.workspace}-clearboard-destination-${var.s3_bucket_unique_suffix}"
  acl    = "private"

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET"]
    allowed_origins = ["*"]
    max_age_seconds = 3600
  }

  tags = {
    Name        = "clearboard-destination"
    Environment = terraform.workspace
  }
}

# Add notification invoking Lambda to enhance an image each time one is
# uploaded to the source bucket.
resource "aws_s3_bucket_notification" "clearboard_source_bucket_notification" {
  bucket = aws_s3_bucket.clearboard_source.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.clearboard_enhance_lambda.arn
    events              = ["s3:ObjectCreated:*"]
  }
}
