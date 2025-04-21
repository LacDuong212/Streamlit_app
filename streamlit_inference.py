import io
from typing import Any
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
import os
import av

# Vô hiệu hóa CUDA để đảm bảo tương thích với Streamlit Cloud
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the Inference class, checking Streamlit requirements and setting up the model path.
        """
        check_requirements("streamlit>=1.29.0")  # Kiểm tra yêu cầu Streamlit

        self.st = st  # Tham chiếu đến module Streamlit
        self.source = None  # Nguồn video (webcam hoặc video file)
        self.enable_trk = False  # Cờ để bật/tắt theo dõi đối tượng
        self.conf = 0.25  # Ngưỡng độ tin cậy cho phát hiện
        self.iou = 0.45  # Ngưỡng IoU cho non-maximum suppression
        self.org_frame = None  # Container cho frame gốc
        self.ann_frame = None  # Container cho frame đã được chú thích
        self.vid_file_name = None  # Tên tệp video hoặc chỉ số webcam
        self.selected_ind = []  # Danh sách chỉ số lớp được chọn
        self.model = None  # Instance của mô hình YOLO
        self.uploaded_file = None  # Tệp video được tải lên

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Ẩn menu chính
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
        self.source = self.st.sidebar.selectbox("Video Source", ("Webcam", "Uploaded Video"))
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))
        self.conf = float(self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

        # Thêm tùy chọn tải lên video
        if self.source == "Uploaded Video":
            self.uploaded_file = self.st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()  # Container cho frame gốc
        self.ann_frame = col2.empty()  # Container cho frame đã chú thích

    def configure(self):
        """Configure the model and load selected classes for inference."""
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

    def inference(self):
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui()
        self.sidebar()
        self.configure()

        if self.st.sidebar.button("Start"):
            stop_button = self.st.button("Stop")

            # Cấu hình WebRTC cho webcam
            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            def video_frame_callback(frame):
                """Xử lý frame video từ webcam hoặc file."""
                img = frame.to_ndarray(format="bgr24")

                # Xử lý frame với mô hình YOLO
                if self.enable_trk == "Yes":
                    results = self.model.track(
                        img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                    )
                else:
                    results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                annotated_frame = results[0].plot()
                self.org_frame.image(img, channels="BGR")
                self.ann_frame.image(annotated_frame, channels="BGR")
                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

            if self.source == "Webcam":
                webrtc_streamer(
                    key="example",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_frame_callback=video_frame_callback
                )
            elif self.source == "Uploaded Video" and self.uploaded_file is not None:
                # Lưu tệp video tạm thời
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(self.uploaded_file.read())
                tfile.close()

                # Mở video bằng OpenCV
                cap = cv2.VideoCapture(tfile.name)
                while cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        break

                    # Xử lý frame với mô hình YOLO
                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                        )
                    else:
                        results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                    annotated_frame = results[0].plot()
                    self.org_frame.image(img, channels="BGR")
                    self.ann_frame.image(annotated_frame, channels="BGR")

                    # Đợi một chút để mô phỏng thời gian thực
                    if stop_button:
                        break
                    cv2.waitKey(30)

                cap.release()
                os.unlink(tfile.name)  # Xóa tệp tạm

if __name__ == "__main__":
    import tempfile
    Inference().inference()
