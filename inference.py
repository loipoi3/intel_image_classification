import torch
from PIL import Image
import albumentations as A
from model import ResNet
from albumentations.pytorch import ToTensorV2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ResNet()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()


def predict(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        prediction = class_labels[predicted.item()]
        return prediction


def preprocess_image(image):
    image = np.array(image)
    transform = A.Compose([A.Resize(64, 64), ToTensorV2()])
    image = transform(image=image)
    image = image['image'].unsqueeze(0)
    image = image / 255.0
    return image


@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']
    image = Image.open(file.stream)
    label = predict(image)
    return jsonify({'class': label})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
