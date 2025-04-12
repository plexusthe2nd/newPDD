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

    st.header("📄 Upload an Image")
    uploaded_image = st.file_uploader("Upload a mango leaf image", type=['jpg', 'jpeg', 'png'])
    if uploaded_image:
        img = Image.open(uploaded_image)
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
elif page == "About":
    st.markdown("""
        <style>
        .hero {
            background-image: url('this is us.jpg');
            background-size: cover;
            background-position: center;
            height: 380px;
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
            text-align: center;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.75);
            border-radius: 10px;
            margin-bottom: 3em;
        }
        .hero h1 {
            font-size: 3em;
            font-weight: bold;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            gap: 2em;
            padding: 0 2em;
            margin-top: 2em;
        }
        .info-card {
            flex: 1;
            min-width: 240px;
            text-align: center;
        }
        .info-card .icon {
            font-size: 2em;
            color: #2e7d32;
            margin-bottom: 0.3em;
        }
        .green-title {
            color: #2e7d32;
            font-weight: 700;
            font-size: 1.3em;
            margin-bottom: 0.4em;
        }
        .para {
            font-size: 1.05em;
            color: #aaa;
            line-height: 1.6;
        }
        </style>

        <div class="hero">
            <h1>We Work for Farmers</h1>
        </div>

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
