import cv2
import numpy as np
from pose import PoseAnalyzer
from gait import GaitAnalyzer
from visualize import create_gait_energy_image, create_symmetry_heatmap, plot_knee_angle_curves, create_symmetry_dashboard

video_path = "videos/crime_scene.mp4"

print("正在分析视频...")
pose = PoseAnalyzer(use_yolo=False)
gait = GaitAnalyzer()

cap = cv2.VideoCapture(video_path)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    landmarks, _ = pose.extract_landmarks(frame)
    if landmarks:
        gait.add_frame(landmarks)
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"已处理 {frame_count} 帧")
cap.release()

if len(gait.keypoints_history) < 10:
    print("检测到的帧数不足，无法生成可视化")
    exit()

print(f"共检测到 {len(gait.keypoints_history)} 帧有效关键点")

# 获取视频尺寸（随便取一帧）
cap = cv2.VideoCapture(video_path)
ret, sample_frame = cap.read()
cap.release()
h, w = sample_frame.shape[:2]

print("生成步态能量图...")
energy_img = create_gait_energy_image(gait.keypoints_history, (h, w))
cv2.imwrite("output/energy.jpg", energy_img)

print("生成对称性热力图...")
symmetry_img = create_symmetry_heatmap(gait.keypoints_history, (h, w))
cv2.imwrite("output/symmetry_heatmap.jpg", symmetry_img)

print("生成膝关节角度曲线...")
plot_knee_angle_curves(gait, "output/knee_curves.png")

print("生成对称性仪表盘...")
results = gait.analyze()
if results.get('symmetry_score'):
    create_symmetry_dashboard(results['symmetry_score'], results.get('recommendation', ''), "output/dashboard.png")
else:
    print("未获取到对称性评分，仪表盘生成失败")

print("可视化结果已保存到 output/ 目录")
