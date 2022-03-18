"""Blackboard enhancer lambda function."""
import boto3

s3_client = boto3.client("s3")

# pylint: disable=unused-argument


def handler(event, context):
    """
    Triggered by S3 each time an image is uploaded to the S3 source bucket.
    Enhance image to extract black board
    and publish the result to S3 destination bucket
    """
    for record in event["Records"]:
        s3_name = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        copy_source = {
            "Bucket": s3_name,
            "Key": key,
        }
        s3_client.copy(copy_source, s3_name.replace("source", "dest"), key)
