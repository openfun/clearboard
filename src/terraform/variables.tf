
variable "cloudfront_price_class" {
  type = map(string)

  default = {
    production = "PriceClass_All"
  }
}

variable "cloudfront_trusted_signer_id" {
  type = string
}

variable "s3_bucket_unique_suffix" {
  type = string
}
