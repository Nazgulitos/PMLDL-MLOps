import streamlit as st
import requests
from PIL import Image

st.title("Blond or Non-blond Hair Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    # Send the image to the FastAPI model for prediction
    if st.button('Predict'):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://api:8000/predict/", files=files)
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error in prediction")
            