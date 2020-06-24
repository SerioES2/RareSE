import mnist_trainer as mt
import datetime as dt
from Utility.DataLoaderAWS import DataLoaderAWS3S

# Training model
mt.RunTrain(True)

# upload training model to S3
now = dt.datetime.now()
time = now.strftime("%Y-%m-%d-%H-%M")
object_name = 'mnist-' + time + '.pth'

s3_client = DataLoaderAWS3S()
s3_client.upload_file('mnist.pth', 'mltraining.20200624', object_name)