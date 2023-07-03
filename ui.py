from flask import Flask, render_template, request
from inference import classify_image
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = request.files['image']
    image_bytes = image.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    prediction = classify_image(image_bytes)
    return render_template('result.html', prediction=prediction, image=image_base64)

if __name__ == '__main__':
    app.run()
