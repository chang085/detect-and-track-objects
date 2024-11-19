# Real-Time Detection and Face Recognition System

## Overview

This project implements a **real-time detection and tracking system** using **YOLO**, **Kalman Filters**, and **MediaPipe** for object detection, tracking, and face recognition. The system identifies humans, tracks their movement, and performs face recognition by comparing detected face embeddings with preloaded reference data. Unrecognized faces are saved for further analysis.

---

## Features

- **YOLO-based Object Detection**: Detects humans in real-time using a pre-trained YOLO model.
- **Kalman Filter Tracking**: Smooths object movements and provides object tracking with prediction.
- **Face Recognition**: 
  - Extracts 3D face embeddings using MediaPipe FaceMesh.
  - Matches detected faces with a reference dataset based on cosine similarity.
- **Unmatched Face Saving**: Stores unmatched faces locally for further processing.
- **Real-Time Visualization**:
  - Draws bounding boxes and IDs for detected humans.
  - Displays "Match" or "Unmatched" status for faces.

---

## Installation

### Requirements

Ensure you have the following dependencies installed:

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- MediaPipe
- SciPy
- Ultralytics YOLO

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>


### Install the necessary Python libraries
``` bash
 pip install opencv-python-headless numpy mediapipe scipy ultralytics
```
# Tải xuống mô hình YOLO ( yolo11n.pt) và đặt nó vào thư mục dự án

# Tạo các thư mục cần thiết
``` bash
mkdir tracked_objects 3d_image_data
```

# Thêm hình ảnh khuôn mặt tham chiếu vào 3d_image_datathư mục để so sánh.

### Cách sử dụng

# Chạy tập lệnh
``` bash
python real_time_tracking.py

```
# Hệ thống sẽ:

Mở nguồn cấp dữ liệu camera thời gian thực.
Phát hiện và theo dõi con người.
Thực hiện nhận dạng khuôn mặt.
Lưu các khuôn mặt không trùng khớp vào tracked_objectsthư mục.
# Nhấn q để thoát khỏi ứng dụng.
## Cấu trúc thư mục
- tracked_objects/: Lưu trữ hình ảnh khuôn mặt không khớp.
- 3d_image_data/: Bao gồm hình ảnh khuôn mặt tham chiếu được sử dụng để so sánh.
# Tùy chỉnh
- Mô hình YOLO : Thay thế yolo11n.ptbằng mô hình được đào tạo trên một tập dữ liệu cụ thể nếu cần.
- Ngưỡng nhận dạng khuôn mặt : Điều chỉnh ngưỡng tương đồng (hiện tại 0.8) trong extract_face_embeddingschức năng để khớp chặt chẽ hơn hoặc lỏng lẻo hơn.
- Giới hạn lưu khuôn mặt : Sửa đổi giới hạn lưu (hiện tại 10) trong logic lưu khuôn mặt không khớp.
## IV. Tổng kết

Dự án này nhằm phát hiện và theo dõi người xuất hiện trong hình trong thời gian thực, đồng thời thực hiện nhận dạng đối tượng . Hệ thống này cấp nhiều tính năng để người dùng có thể theo dõi và nhận dạng đối tượng. Nếu bạn có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ với chúng tôi qua email: 22010085@st.phenikaa-uni.edu.vn
