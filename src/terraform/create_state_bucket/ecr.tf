resource "aws_ecr_repository" "clearboard_lambda" {
  name                 = "clearboard-lambda"
  image_tag_mutability = "MUTABLE"
}

