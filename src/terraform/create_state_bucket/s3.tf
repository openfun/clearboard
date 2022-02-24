resource "aws_kms_key" "state_key" {
  description = "Used by Terraform to store remote state for the FUN infra project"
}

resource "aws_s3_bucket" "state_bucket" {
  bucket = "clearboard-terraform-${var.s3_bucket_unique_suffix}"

  tags = {
    Name = "terraform"
  }
}

resource "aws_s3_bucket_acl" "state_bucket_acl" {
  bucket = aws_s3_bucket.state_bucket.id
  acl    = "private"
}

resource "aws_s3_bucket_versioning" "state_bucket_versioning" {
  bucket = aws_s3_bucket.state_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state_bucket_sse" {
  bucket = aws_s3_bucket.state_bucket.bucket
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = "${aws_kms_key.state_key.arn}"
      sse_algorithm     = "aws:kms"
    }
  }
}
