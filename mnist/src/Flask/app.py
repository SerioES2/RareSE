from flask import Flask, jsonify, render_template, request

app = Flask(__name__, static_folder='./images/')

@app.route('/')
def index():
  return render_template('index.html')

if __name__ == '__main__':
  app.run()

