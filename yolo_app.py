import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from yolo_detect import run_detection

# --- PAGE SETUP ---
st.set_page_config(page_title="Mango Health Scanner", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'my_model.pt')

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🍃 Mango Health Scanner")
page = st.sidebar.radio("Go to", ["Home", "Detection", "About"])

# --- HOME PAGE ---
if page == "Home":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
            <h1 style="color: #2b7a0b; font-size: 2.8em;">A Look Into the Future of Agriculture</h1>
            <p style="font-size: 1.1em;">This tool helps you scan and analyze mango leaves using real-time AI object detection.
            It's fast, simple, and powerful — built for farmers, researchers, and agriculture enthusiasts.</p>
            <p style="margin-top: 2em; font-size: 0.9em; color: #777;">© 2025 RIM17A. All rights reserved.</p>
        """, unsafe_allow_html=True)

    with col2:
        try:
            image_path = "Webpic0.jpg"
            image = Image.open(image_path)
            st.image(image, use_column_width=True, caption="Mango field photo")
        except:
            st.info("No image found. You can add one named 'Webpic0.jpg' in the app folder.")

# --- DETECTION PAGE ---
elif page == "Detection":
    st.title("🍃 Mango Leaf Detection")

    st.header(" Upload an Image")
    uploaded_image = st.file_uploader("Upload a mango leaf image", type=['jpg', 'jpeg', 'png'])
    if uploaded_image:
        img = Image.open(uploaded_image)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = run_detection(model_path, img_cv, threshold=0.5)
        processed_img, detected_classes = results[0]
        st.image(processed_img, channels="BGR", caption="Detection Result")

        if detected_classes:
            st.subheader(" Detected Classes:")
            for c in sorted(set(detected_classes)):
                st.markdown(f"- **{c}**")
        else:
            st.info("No mango leaf issue detected.")

    st.header("📸 Capture from Mobile Camera")
    cam_image = st.camera_input("Take a picture")
    if cam_image:
        img = Image.open(cam_image)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = run_detection(model_path, img_cv, threshold=0.5)
        processed_img, detected_classes = results[0]
        st.image(processed_img, channels="BGR", caption="Detection Result")

        if detected_classes:
            st.subheader("🧠 Detected Classes:")
            for c in sorted(set(detected_classes)):
                st.markdown(f"- **{c}**")
        else:
            st.info("No mango leaf issue detected.")

    st.header("🎥 Real-time Detection (Webcam)")

    class YOLOProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            results = run_detection(model_path, img, threshold=0.5)
            processed_img, _ = results[0]
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    webrtc_streamer(key="yolo-stream", video_processor_factory=YOLOProcessor, async_processing=False)

# --- ABOUT PAGE ---
# --- ABOUT PAGE ---
elif page == "About":
    col1, col2, col3 = st.columns([0.2, 1, 0.2])
    with col2:
        st.image("this is us.jpg", use_column_width=True, caption="We Work for Farmers")

    st.markdown("""
        <style>
        .info-row {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 2em;
            margin-top: 3em;
        }
        .info-card {
            flex: 1;
            min-width: 250px;
            background: #f9f9f9;
            padding: 1.5em;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.07);
            text-align: center;
        }
        .icon {
            font-size: 2em;
            margin-bottom: 0.4em;
            color: #2e7d32;
        }
        .green-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #2e7d32;
            margin-bottom: 0.3em;
        }
        .para {
            font-size: 1.05em;
            color: #555;
            line-height: 1.6;
        }
        </style>

        <div class="info-row">
            <div class="info-card">
                <div class="icon">📌</div>
                <div class="green-title">What We Stand For</div>
                <p class="para">
                    We help farmers, students, and agriculture lovers detect mango leaf diseases early with computer vision.
                </p>
            </div>

            <div class="info-card">
                <div class="icon">🎯</div>
                <div class="green-title">Our Mission</div>
                <p class="para">
                    Make AI-powered mango disease detection accessible and easy to use.
                </p>
            </div>

            <div class="info-card">
                <div class="icon">📱</div>
                <div class="green-title">Our Vision</div>
                <p class="para">
                    A future where farmers scan leaves with a phone and get real-time results.
                </p>
            </div>
        </div>

        <p style='text-align: center; font-size: 14px; color: gray; margin-top: 4em;'>
            © 2025 RIM17A. All rights reserved.
        </p>
    """, unsafe_allow_html=True)



