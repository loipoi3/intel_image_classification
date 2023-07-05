import streamlit as st
import requests
from PIL import Image
import io

def send_prediction_request(image):
    url = 'http://api:5000/predict'
    image_byte_arr = io.BytesIO()
    image.save(image_byte_arr, format='JPEG')
    image_byte_arr = image_byte_arr.getvalue()
    files = {'file': image_byte_arr}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        result = response.json()
        return result.get('class')
    else:
        return 'Error'


def main():
    st.title("Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        label = send_prediction_request(image)
        st.write(f"Class: {label}")

if __name__ == '__main__':
    st.set_page_config(page_title="Image Classification")
    main()
