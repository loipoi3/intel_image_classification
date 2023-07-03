import torch
from torchvision import transforms
from PIL import Image
import albumentations as A
from model import ResNet
from albumentations.pytorch import ToTensorV2
import numpy as np
from io import BytesIO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model
model = ResNet()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

def preprocess_image(image):
    image = np.array(image).astype(np.uint8)
    transform = A.Compose([A.Resize(64, 64), ToTensorV2()])
    image = transform(image=image)['image'].unsqueeze(0).float()
    return image

def classify_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        prediction = class_labels[predicted.item()]

    return prediction
