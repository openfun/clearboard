output "cloudfront_domain" {
  value = aws_cloudfront_distribution.clearboard_cloudfront_distribution.domain_name
}

output "iam_trusted_signer_access_key_id" {
  value = aws_iam_access_key.clearboard_access_key.id
}

output "iam_secret_access_key" {
  value = aws_iam_access_key.clearboard_access_key.secret
  sensitive = true
}
