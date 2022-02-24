# Define a user and associate appropriate policies
resource "aws_iam_user" "clearboard_user" {
  name = "${terraform.workspace}-clearboard"
}

resource "aws_iam_access_key" "clearboard_access_key" {
  user = aws_iam_user.clearboard_user.name
}

# Grant user access to the source bucket
resource "aws_s3_bucket_policy" "clearboard_source_bucket_policy" {
  bucket = aws_s3_bucket.clearboard_source.id

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "User access",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${aws_iam_user.clearboard_user.arn}"
      },
      "Action": [ "s3:*" ],
      "Resource": [
        "${aws_s3_bucket.clearboard_source.arn}",
        "${aws_s3_bucket.clearboard_source.arn}/*"
      ]
    }
  ]
}
EOF
}

# Grant user access to the destination bucket
resource "aws_s3_bucket_policy" "clearboard_destination_bucket_policy" {
  bucket = aws_s3_bucket.clearboard_destination.id

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "User access",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${aws_iam_user.clearboard_user.arn}"
      },
      "Action": [ "s3:*" ],
      "Resource": [
        "${aws_s3_bucket.clearboard_destination.arn}",
        "${aws_s3_bucket.clearboard_destination.arn}/*"
      ]
    },
    {
      "Sid": "Cloudfront",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${aws_cloudfront_origin_access_identity.clearboard_oai.iam_arn}"
      },
      "Action": "s3:GetObject",
      "Resource": "${aws_s3_bucket.clearboard_destination.arn}/*"
    }
  ]
}
EOF
}
