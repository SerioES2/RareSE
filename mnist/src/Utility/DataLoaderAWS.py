import boto3
import logging
from botocore.exceptions import ClientError

class DataLoaderAWSS3:

    def __init__(self):
        self.__s3_client = boto3.client('s3')

    def upload_file(self, file_name, bucket, object_name=None):
        if object_name is None:
            object_name = file_name

        try:
            self.__s3_client.upload_file(file_name, bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def download_file(self, file_name, bucket, object_name=None):
        if object_name is None:
            object_name = file_name

        try:
            self.__s3_client.download_file(bucket, object_name, file_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

