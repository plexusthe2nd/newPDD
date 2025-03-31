import streamlit as st
import cv2
from yolo_detect import run_detection
from PIL import Image
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'my_model.pt')

st.title("YOLOv8 Object Detection")

# Upload Image Section
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

# Camera Input for Mobile Users
st.write("## Capture Image from Mobile Camera")
captured_image = st.camera_input("Take a picture")

if captured_image is not None:
    img = Image.open(captured_image)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    temp_path = 'temp_mobile.jpg'
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

# Real-time Video Processing with streamlit-webrtc
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Resize frame for faster processing
        img_resized = cv2.resize(img, (640, 640))

        # Save temp frame
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, img_resized)

        # Run detection
        results = run_detection(model_path, temp_path, threshold=0.5)
        processed_frame, _ = results[0]

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

st.write("## Real-time Object Detection (PC & Mobile)")
webrtc_streamer(
    key="yolo-stream",
    video_processor_factory=YOLOProcessor,
    async_processing=False  # Ensures smoother real-time detection
)
