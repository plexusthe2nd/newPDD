# yolo_inference.py

import os
import sys
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Set default model + color scheme
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

def run_detection(model_path, source_path, threshold=0.5, resolution=None, max_frames=None):
    """
    Run YOLOv8 object detection on image, video, or webcam source.
    Returns list of processed frames (with boxes drawn).
    """
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at: {model_path}')
    
    model = YOLO(model_path, task='detect')
    labels = model.names

    # Parse resolution
    if resolution:
        resW, resH = map(int, resolution.lower().split('x'))
        resize = True 
    else:
        resW, resH = 640, 480  # default resolution
        resize = False
        


    # Determine source type
    if os.path.isdir(source_path):
        source_type = 'folder'
    elif os.path.isfile(source_path):
        ext = os.path.splitext(source_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            source_type = 'image'
        elif ext in ['.avi', '.mov', '.mp4', '.mkv', '.wmv']:
            source_type = 'video'
        else:
            raise ValueError('Unsupported file type.')
    elif 'usb' in source_path:
        source_type = 'usb'
        usb_idx = int(source_path[3:])
    else:
        raise ValueError('Invalid source input.')

    processed_frames = []
    frame_count = 0

    if source_type in ['image', 'folder']:
        if source_type == 'image':
            imgs_list = [source_path]
        else:
            imgs_list = [f for f in glob.glob(os.path.join(source_path, '*')) 
                         if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

        for img_path in imgs_list:
            frame = cv2.imread(img_path)
            processed = process_frame(model, frame, labels, threshold, resize, resW, resH)
            processed_frames.append(processed)
    
    elif source_type in ['video', 'usb']:
        cap = cv2.VideoCapture(source_path if source_type == 'video' else usb_idx)

        if resize:
            cap.set(3, resW)
            cap.set(4, resH)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = process_frame(model, frame, labels, threshold, resize, resW, resH)
            processed_frames.append(processed)

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()

    return processed_frames


def process_frame(model, frame, labels, threshold=0.5, resize=False, resW=640, resH=480):
    """
    Processes a single frame with YOLOv8 model and draws detections.
    Returns the processed frame.
    """
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > threshold:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count += 1

    return frame
