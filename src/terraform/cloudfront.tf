locals {
  s3_destination_origin_id = "clearboard-destination-origin"
}

# Create an origin access identity that will allow CloudFront to access S3
# See bucket policies in s3.tf or documentation for more details:
# https://www.terraform.io/docs/providers/aws/r/cloudfront_origin_access_identity.html
resource "aws_cloudfront_origin_access_identity" "clearboard_oai" {
  comment = "Clearboard origin for the ${terraform.workspace} environment"
}

resource "aws_cloudfront_distribution" "clearboard_cloudfront_distribution" {
  # Origin for the destination S3 bucket
  origin {
    domain_name = aws_s3_bucket.clearboard_destination.bucket_domain_name
    origin_id   = local.s3_destination_origin_id

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.clearboard_oai.cloudfront_access_identity_path
    }
  }

  enabled             = true
  is_ipv6_enabled     = true
  wait_for_deployment = false

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = local.s3_destination_origin_id
    trusted_signers  = [var.cloudfront_trusted_signer_id]

    forwarded_values {
      query_string = false
      headers = ["Access-Control-Request-Headers", "Access-Control-Request-Method", "Origin"]

      cookies {
        forward = "none"
      }
    }

    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
  }

  price_class = lookup(var.cloudfront_price_class, terraform.workspace, "PriceClass_100")

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = {
    Environment = terraform.workspace
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
