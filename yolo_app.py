# streamlit_app.py
import streamlit as st
import cv2
from yolo_detect import run_detection
from PIL import Image
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'my_model.pt')


# Load model + image
uploaded_image = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])

if uploaded_image is not None:
    file_bytes = uploaded_image.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save to temp file
    temp_path = 'temp_input.jpg'
    cv2.imwrite(temp_path, img)

    results = run_detection(model_path, temp_path, threshold=0.5)

    # Display result
    st.image(results[0], channels="BGR", caption="Detection result")
