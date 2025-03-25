import streamlit as st
import cv2
from yolo_detect import run_detection
from PIL import Image
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'my_model.pt')

st.title("YOLOv8 Object Detection")
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    file_bytes = uploaded_image.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    temp_path = 'temp_input.jpg'
    cv2.imwrite(temp_path, img)

    results = run_detection(model_path, temp_path, threshold=0.5)

    processed_img, detected_classes = results[0]

    st.image(processed_img, channels="BGR", caption="Detection result")

    if detected_classes:
        st.write("### Detected Classes:")
        for c in sorted(set(detected_classes)):
            st.markdown(f"- **{c}**")
    else:
        st.info("No mango detected.")
