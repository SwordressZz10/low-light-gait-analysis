import cv2
import torch
import numpy as np
import mediapipe as mp

class PoseAnalyzer:
    """人体姿态分析器（YOLOv5 + MediaPipe）"""
    
    def __init__(self, use_yolo=True):
        self.use_yolo = use_yolo
        self.yolo_model = None
        
        # 初始化 MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 关键点名称映射
        self.keypoints = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        }
        
        # 加载 YOLOv5 模型
        if self.use_yolo:
            self._load_yolo()
    
    def _load_yolo(self):
        """加载 YOLOv5 模型"""
        try:
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.yolo_model.conf = 0.5
            self.yolo_model.classes = [0]
            print("✅ YOLOv5 人体检测器已加载")
        except Exception as e:
            print(f"⚠️ YOLOv5 加载失败: {e}")
            self.use_yolo = False
    
    def detect_person(self, image):
        """YOLOv5检测人体，返回边界框"""
        if not self.use_yolo or self.yolo_model is None:
            return None
        
        results = self.yolo_model(image)
        detections = results.pandas().xyxy[0]
        
        if len(detections) == 0:
            return None
        
        best = detections.iloc[0]
        x1, y1, x2, y2 = int(best['xmin']), int(best['ymin']), int(best['xmax']), int(best['ymax'])
        return (x1, y1, x2, y2)
    
    def extract_landmarks(self, image):
        """提取姿态关键点"""
        h_img, w_img = image.shape[:2]
        bbox = self.detect_person(image)
        
        if bbox is not None and self.use_yolo:
            x1, y1, x2, y2 = bbox
            person_roi = image[y1:y2, x1:x2]
            if person_roi.size == 0:
                return self._extract_mediapipe(image)
            
            h_roi, w_roi = person_roi.shape[:2]
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_roi)
            
            if results.pose_landmarks:
                landmarks = {}
                for name, idx in self.keypoints.items():
                    lm = results.pose_landmarks.landmark[idx]
                    x_pixel = lm.x * w_roi + x1
                    y_pixel = lm.y * h_roi + y1
                    x = x_pixel / w_img
                    y = y_pixel / h_img
                    landmarks[name] = (x, y, lm.visibility)
                return landmarks, results.pose_landmarks
        
        return self._extract_mediapipe(image)
    
    def _extract_mediapipe(self, image):
        """直接使用 MediaPipe 提取关键点"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        landmarks = {}
        if results.pose_landmarks:
            for name, idx in self.keypoints.items():
                lm = results.pose_landmarks.landmark[idx]
                landmarks[name] = (lm.x, lm.y, lm.visibility)
        
        return landmarks, results.pose_landmarks
    
    def draw_landmarks(self, image, pose_landmarks):
        if pose_landmarks:
            self.mp_draw.draw_landmarks(
                image, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        return image


def calculate_angle(a, b, c):
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1, 1)) * 180 / np.pi
    return angle
