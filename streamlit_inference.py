# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any
import cv2
import numpy as np
from PIL import Image
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Streamlit and WebRTC imports at module level
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Patch Streamlit's file watcher to avoid PyTorch module path extraction issue
import streamlit.watcher.local_sources_watcher
from streamlit.watcher.local_sources_watcher import extract_paths

def patched_extract_paths(module):
    try:
        return extract_paths(module)
    except Exception:
        return []

streamlit.watcher.local_sources_watcher.extract_paths = patched_extract_paths

class Inference:
    """
    A class to perform object detection, image classification, image segmentation, and pose estimation inference.

    This class provides functionalities for loading models, configuring settings, uploading video files, and performing
    real-time inference using Streamlit, streamlit_webrtc, and Ultralytics YOLO models.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the Inference class, checking requirements and setting up the model path.

        Args:
            **kwargs (Any): Additional keyword arguments for model configuration.
        """
        check_requirements(["streamlit>=1.29.0", "streamlit-webrtc>=0.47.7"])
        self.st = st
        self.webrtc_streamer = webrtc_streamer
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = self.temp_dict.get("model")
        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""

        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")
        self.source = self.st.sidebar.selectbox("Video", ("webcam", "video"))
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))
        self.conf = float(self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        """Handle video file uploads through the Streamlit interface."""
        self.vid_file_name = None
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("ultralytics.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "ultralytics.mp4"

    def configure(self):
        """Configure the model and load selected classes for inference."""
        from ultralytics import YOLO  # Lazy import to avoid early PyTorch issues
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")
            class_names = list(self.model.names.values())
        self.st.success("Model loaded successfully!")

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]
        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    class VideoProcessor(VideoProcessorBase):
        """A class to process video frames for streamlit_webrtc."""
        def __init__(self, model, enable_trk, conf, iou, selected_ind):
            self.model = model
            self.enable_trk = enable_trk
            self.conf = conf
            self.iou = iou
            self.selected_ind = selected_ind

        def recv(self, frame):
            """Process incoming webcam frames."""
            img = frame.to_ndarray(format="bgr24")  # Convert WebRTC frame to BGR

            # Perform inference
            if self.enable_trk == "Yes":
                results = self.model.track(
                    img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                )
            else:
                results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)

            annotated_frame = results[0].plot()  # Add annotations
            return annotated_frame

    def inference(self):
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        if self.st.sidebar.button("Start"):
            stop_button = self.st.button("Stop")

            if self.source == "video" and self.vid_file_name:
                cap = cv2.VideoCapture(self.vid_file_name)
                if not cap.isOpened():
                    self.st.error("Could not open video file.")
                    return

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        self.st.warning("Failed to read frame from video.")
                        break

                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                        )
                    else:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                    annotated_frame = results[0].plot()

                    if stop_button:
                        cap.release()
                        self.st.stop()

                    self.org_frame.image(frame, channels="BGR")
                    self.ann_frame.image(annotated_frame, channels="BGR")

                cap.release()

            elif self.source == "webcam":
                ctx = self.webrtc_streamer(
                    key="webcam",
                    video_processor_factory=lambda: self.VideoProcessor(
                        self.model, self.enable_trk, self.conf, self.iou, self.selected_ind
                    ),
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                )

                if ctx.video_processor:
                    self.org_frame.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")
                    self.ann_frame.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")

                if stop_button:
                    self.st.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None
    Inference(model=model).inference()
