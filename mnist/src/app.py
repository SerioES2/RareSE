from flask import Flask, jsonify, render_template, request
import mnist_prediction as mp 

prediction_model = mp.load_prediction_model('../model/mnist.pth')

app = Flask(__name__, static_folder='./images/')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  image_path = request.form["file"]
  image_data = mp.read_image(('./images/' + image_path))
  prediction = mp.predict(prediction_model, image_data)
  return render_template('predict.html', answer = prediction.item(), image_file=request.form["file"], image_path = image_path)

if __name__ == '__main__':
  app.run()

