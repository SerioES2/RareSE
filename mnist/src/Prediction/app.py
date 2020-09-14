import ast
import json
import base64

from io import BytesIO
from flask import Flask, jsonify, request
from PIL import Image

import mnist_prediction as mp 
#from Utility import DataLoaderAWS as aws

app = Flask(__name__, static_folder='./tmp_image')

predictor = None

def load_model():
  global predictor
#
#  # download training model to S3
#  s3_client = aws.DataLoaderAWSS3()
#  s3_client.download_file('mnist.pth', 'mltraining.20200624', 'mnist-2020-06-24-11-19.pth')
#
  # prediction
  predictor = mp.Predictor()
  predictor.LoadModel('mnist.pth')

# str >> base64 >> bytes >> image
def convert_base64Str_to_Image(base64str):
  # str >> base64
  img = base64.b64decode(base64str)
  # base64 >> bytes
  img = BytesIO(img)
  # bytes >> image
  img = Image.open(img)

  return img

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  response = {
    'Success' : False,
  }

  print("get post message")

  if request.method == 'POST':

    json_data = request.get_json()
    if type(json_data) == str :
      json_data = json.loads(json_data)

    img = convert_base64Str_to_Image(json_data['feature'])

    predicted = predictor.Predict2(img)

    response['Success'] = True
    response['Prediction'] = predicted.item()
    print('----Prediction result----')
    print(response)
    print('----------------------')
  
  respJson = jsonify(response) 
  respJson.status_code = 200
  return respJson

if __name__ == '__main__':
  load_model()
  print('Flask starting server...')
  app.run(host='0.0.0.0', port=5000)


