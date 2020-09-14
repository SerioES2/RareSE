import requests
import json
import base64
from io import BytesIO
from PIL import Image


def get_json_data():
    img = Image.open("mnist_0.png")

    # convert Image to bytes to base64
    # Image >> bytes
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    # bytes >> base64
    img_base64 = base64.b64encode(img_byte)
    img_str = img_base64.decode('utf-8')

    jsonData = {
        "text" : "sample",
        "feature" : img_str
    }

    return jsonData

url = 'http://192.168.20.10:8081/predict'
r = requests.post(url, json=json.dumps(get_json_data()))