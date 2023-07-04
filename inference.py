import torch
import streamlit as st
from PIL import Image
import albumentations as A
from model import ResNet
from albumentations.pytorch import ToTensorV2
import numpy as np


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

def main():
    st.title("Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        label = predict(image)
        st.write(f"Class: {label}")


if __name__ == '__main__':
    main()
