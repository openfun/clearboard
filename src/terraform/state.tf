
terraform {
  backend "s3" {
    key            = "clearboard.tfstate"
    dynamodb_table = "clearboard_terraform_state_locks"
    encrypt        = true
  }
}
