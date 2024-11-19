import cv2
import numpy as np
import os
from ultralytics import YOLO
import mediapipe as mp
from scipy.spatial.distance import cosine

class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.predicted = None

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)
        self.predicted = self.kalman.predict()
        return self.predicted

# Tạo thư mục lưu trữ
if not os.path.exists("tracked_objects"):
    os.makedirs("tracked_objects")
if not os.path.exists("3d_image_data"):
    os.makedirs("3d_image_data")

# Load YOLO model (chỉ nhận diện người)
model = YOLO('yolo11n.pt')  # Thay bằng đường dẫn mô hình YOLO
model.classes = [0]  # Chỉ nhận diện người (class ID = 0)

# Khởi tạo MediaPipe Face Detection và FaceMesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Hàm tạo embeddings từ ảnh khuôn mặt
def extract_face_embeddings(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(face_rgb)
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()
    return None

# Hàm tải embeddings từ thư mục 3d_image_data
def load_reference_embeddings(reference_dir="3d_image_data"):
    reference_embeddings = {}
    for filename in os.listdir(reference_dir):
        file_path = os.path.join(reference_dir, filename)
        if filename.endswith((".jpg", ".png")):
            img = cv2.imread(file_path)
            embeddings = extract_face_embeddings(img)
            if embeddings is not None:
                reference_embeddings[filename] = embeddings
    return reference_embeddings

reference_embeddings = load_reference_embeddings()

# Mở camera
cap = cv2.VideoCapture(0)
trackers = {}
saved_images_count = {}
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame, stream=True)

    detections = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            if int(cls) == 0:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detections.append((cx, cy, x1, y1, x2, y2))

    updated_trackers = {}
    for i, (cx, cy, x1, y1, x2, y2) in enumerate(detections):
        if i not in trackers:
            trackers[i] = KalmanTracker()
        predicted = trackers[i].update(cx, cy)
        updated_trackers[i] = trackers[i]
        pred_x, pred_y = int(predicted[0]), int(predicted[1])

        # Vẽ khung và vị trí
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"ID: {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Nhận diện khuôn mặt
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size > 0:
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(person_rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = person_roi.shape
                    fx, fy, fw, fh = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    face_img = person_roi[fy:fy + fh, fx:fx + fw]

                    if fx + fw > iw or fy + fh > ih:
                        print("Invalid face ROI, skipping.")
                        continue

                    face_img = person_roi[fy:fy + fh, fx:fx + fw]

                    if face_img is None or face_img.size == 0:
                        print("Face image is empty, skipping.")
                        continue
                    # So sánh khuôn mặt
                    face_embeddings = extract_face_embeddings(face_img)
                    matched = False
                    if face_embeddings is not None:
                        for ref_name, ref_embedding in reference_embeddings.items():
                            similarity = 1 - cosine(face_embeddings, ref_embedding)
                            if similarity > 0.8:
                                cv2.putText(frame, f"Match: {ref_name}", (x1, y1 - 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                matched = True
                                break

                    if not matched:
                        obj_id = f"unmatched_{x1}_{y1}_{x2}_{y2}"
                        if obj_id not in saved_images_count:
                            saved_images_count[obj_id] = 0

                        if saved_images_count[obj_id] < 10:
                            unmatched_roi = frame[y1:y2, x1:x2]
                            face_path = f"tracked_objects/{obj_id}_{saved_images_count[obj_id]}.jpg"
                            cv2.imwrite(face_path, unmatched_roi)
                            saved_images_count[obj_id] += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Unmatched", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    trackers = updated_trackers
    cv2.imshow("Real-Time Detection and Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
