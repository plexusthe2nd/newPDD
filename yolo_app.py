import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    model = YOLO("my_model.pt")
    return model

model = load_model()
labels = model.names
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

st.title("YOLOv8 Object Detection Web App")
st.write("Available classes:", list(labels.values()))

mode = st.radio("Choose input mode", ["Upload Image", "Use Webcam"])

def run_inference(frame):
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    detected_classes = []

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        conf = detections[i].conf.item()
        classidx = int(detections[i].cls.item())
        classname_raw = labels[classidx]

        if conf > 0.5:
            object_count += 1
            detected_classes.append(classname_raw)
            color = bbox_colors[classidx % 10]
            label = f'{classname_raw}: {int(conf*100)}%'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, object_count, list(set(detected_classes))

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        processed_frame, object_count, detected_classes = run_inference(frame)
        st.image(processed_frame, channels="RGB", caption=f"Detected {object_count} object(s)", use_column_width=True)
        st.write("Detected classes:", detected_classes)

elif mode == "Use Webcam":
    st.write("Press 'Start' to begin webcam detection (limited to short sessions).")
    run = st.checkbox("Start Camera")
    
    if run:
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access camera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, object_count, detected_classes = run_inference(frame)
            frame_window.image(processed_frame, channels="RGB")
            st.write("Detected classes:", detected_classes)
        
        cap.release()
