# streamlit_app.py
import streamlit as st
import cv2
from yolo_detect import run_detection
from PIL import Image
import numpy as np

# Load model + image
model_path = 'runs/detect/train/weights/best.pt'
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
