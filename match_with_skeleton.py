import os
import cv2
import numpy as np
import joblib
import re
from collections import defaultdict
from pose import PoseAnalyzer
from gait import GaitAnalyzer
from enhance import enhance_lowlight

def calculate_angle(a, b, c):
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi

def find_peaks(arr):
    peaks = []
    for i in range(1, len(arr)-1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            peaks.append(i)
    return peaks

def extract_features_improved(video_path, pose, gait):
    cap = cv2.VideoCapture(video_path)
    gait.clear()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = enhance_lowlight(frame)
        landmarks, _ = pose.extract_landmarks(frame)
        if landmarks:
            gait.add_frame(landmarks)
    cap.release()
    
    if len(gait.keypoints_history) < 10:
        return None
    
    joints = ['left_shoulder','right_shoulder','left_elbow','right_elbow',
              'left_wrist','right_wrist','left_hip','right_hip',
              'left_knee','right_knee','left_ankle','right_ankle']
    
    features = []
    for joint in joints:
        x_vals, y_vals, vis_vals = [], [], []
        for kp in gait.keypoints_history:
            if joint in kp:
                x, y, vis = kp[joint]
                x_vals.append(x)
                y_vals.append(y)
                vis_vals.append(vis)
        if len(x_vals) > 0 and np.mean(vis_vals) > 0.3:
            features.extend([np.mean(x_vals), np.std(x_vals), np.mean(y_vals), np.std(y_vals)])
        else:
            features.extend([0,0,0,0])
    
    # 膝关节角度特征
    left_angles, right_angles = [], []
    for kp in gait.keypoints_history:
        if 'left_hip' in kp and 'left_knee' in kp and 'left_ankle' in kp:
            ang = calculate_angle(kp['left_hip'], kp['left_knee'], kp['left_ankle'])
            left_angles.append(ang)
        if 'right_hip' in kp and 'right_knee' in kp and 'right_ankle' in kp:
            ang = calculate_angle(kp['right_hip'], kp['right_knee'], kp['right_ankle'])
            right_angles.append(ang)
    
    if left_angles and right_angles:
        features.extend([
            np.mean(left_angles), np.std(left_angles),
            np.mean(right_angles), np.std(right_angles),
            100 - abs(np.mean(left_angles) - np.mean(right_angles)) * 2,
            np.max(left_angles) - np.min(left_angles),
            np.max(right_angles) - np.min(right_angles),
        ])
    else:
        features.extend([0]*7)
    
    # 步频特征
    left_y = [kp['left_ankle'][1] for kp in gait.keypoints_history if 'left_ankle' in kp]
    right_y = [kp['right_ankle'][1] for kp in gait.keypoints_history if 'right_ankle' in kp]
    
    left_peaks = find_peaks(left_y) if len(left_y) > 5 else []
    right_peaks = find_peaks(right_y) if len(right_y) > 5 else []
    
    features.extend([
        len(left_peaks) / (len(gait.keypoints_history) + 1e-6),
        len(right_peaks) / (len(gait.keypoints_history) + 1e-6),
        (len(left_peaks) - len(right_peaks)) / (len(gait.keypoints_history) + 1e-6),
    ])
    
    return np.array(features)

def draw_force_lines(frame, landmarks, h, w):
    """绘制力线（骨架 + 关节角度标注）"""
    if not landmarks:
        return frame
    
    # 关键点连接（力线）
    connections = [
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ]
    
    # 获取像素坐标
    points = {}
    for name, (x, y, vis) in landmarks.items():
        px, py = int(x * w), int(y * h)
        points[name] = (px, py, vis)
    
    # 绘制力线
    for start, end in connections:
        if start in points and end in points:
            x1, y1, v1 = points[start]
            x2, y2, v2 = points[end]
            if v1 > 0.3 and v2 > 0.3:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 绘制关键点
    for name, (px, py, vis) in points.items():
        if vis > 0.3:
            color = (0, 255, 0) if 'right' in name else (0, 200, 0)
            cv2.circle(frame, (px, py), 4, color, -1)
    
    # 计算并显示膝关节角度
    if 'left_hip' in points and 'left_knee' in points and 'left_ankle' in points:
        _, _, v1 = points['left_hip']
        _, _, v2 = points['left_knee']
        _, _, v3 = points['left_ankle']
        if v1 > 0.3 and v2 > 0.3 and v3 > 0.3:
            a = np.array(points['left_hip'][:2])
            b = np.array(points['left_knee'][:2])
            c = np.array(points['left_ankle'][:2])
            angle = calculate_angle(a, b, c)
            knee_pos = points['left_knee'][:2]
            cv2.putText(frame, f"{angle:.0f}", (knee_pos[0]-20, knee_pos[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    if 'right_hip' in points and 'right_knee' in points and 'right_ankle' in points:
        _, _, v1 = points['right_hip']
        _, _, v2 = points['right_knee']
        _, _, v3 = points['right_ankle']
        if v1 > 0.3 and v2 > 0.3 and v3 > 0.3:
            a = np.array(points['right_hip'][:2])
            b = np.array(points['right_knee'][:2])
            c = np.array(points['right_ankle'][:2])
            angle = calculate_angle(a, b, c)
            knee_pos = points['right_knee'][:2]
            cv2.putText(frame, f"{angle:.0f}", (knee_pos[0]+5, knee_pos[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return frame

print("="*60)
print("匹配嫌疑人 + 骨架力线分析（带骨架版）")
print("="*60)

model = joblib.load("gait_model_final.pkl")
pose = PoseAnalyzer(use_yolo=False)
gait = GaitAnalyzer()

# 分析犯罪现场
crime_feat = extract_features_improved("videos/crime_scene.mp4", pose, gait)
if crime_feat is None:
    print("犯罪现场特征提取失败")
    exit()

# 分析所有嫌疑人
groups = defaultdict(list)

for f in os.listdir("videos"):
    if f.startswith("suspect_") and f.endswith(".mp4"):
        suspect_feat = extract_features_improved(f"videos/{f}", pose, gait)
        if suspect_feat is None:
            continue
        
        diff = np.abs(crime_feat - suspect_feat)
        prod = crime_feat * suspect_feat
        pair = np.concatenate([diff, prod, crime_feat, suspect_feat]).reshape(1, -1)
        prob = model.predict_proba(pair)[0][1]
        
        match = re.search(r'suspect_(\d+)', f)
        sid = match.group(1) if match else f
        groups[sid].append(prob)
        print(f"{f}: {prob*100:.1f}%")

# 计算平均值
results = []
for sid, probs in groups.items():
    avg_prob = np.mean(probs)
    results.append({"id": sid, "avg_prob": avg_prob, "count": len(probs)})

results.sort(key=lambda x: x["avg_prob"], reverse=True)

print("\n" + "="*60)
print("匹配结果（同一嫌疑人取平均值）")
print("="*60)

for r in results:
    if r["avg_prob"] > 0.7:
        tag = "✅ 高度匹配"
    elif r["avg_prob"] > 0.5:
        tag = "⚠️ 可能匹配"
    else:
        tag = "❌ 不匹配"
    print(f"嫌疑人 {r['id']}: {r['avg_prob']*100:.1f}% (基于 {r['count']} 个视频) {tag}")

if results:
    best = results[0]
    print(f"\n🏆 最佳匹配: 嫌疑人 {best['id']} (平均 {best['avg_prob']*100:.1f}%)")
    
    # 使用 normal 版本生成视频
    best_file = f"suspect_{best['id']}_normal.mp4"
    if not os.path.exists(f"videos/{best_file}"):
        for f in os.listdir("videos"):
            if f.startswith(f"suspect_{best['id']}") and f.endswith(".mp4"):
                best_file = f
                break
    
    print(f"\n生成对比视频（带骨架 + 力线分析）: {best_file}")
    
    cap_best = cv2.VideoCapture(f"videos/{best_file}")
    cap_crime = cv2.VideoCapture("videos/crime_scene.mp4")
    
    fps = int(cap_best.get(cv2.CAP_PROP_FPS))
    w1, h1 = int(cap_best.get(3)), int(cap_best.get(4))
    w2, h2 = int(cap_crime.get(3)), int(cap_crime.get(4))
    target_w_single = 640
target_h_single = 360
w1n = target_w_single
h1n = target_h_single
w2n = target_w_single
h2n = target_h_single
# target_h = 480
    w1n = int(w1 * target_h / h1)
    w2n = int(w2 * target_h / h2)
    
    out = cv2.VideoWriter("output/comparison_with_skeleton.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1n + w2n, target_h))
    
    frame_count = 0
    while True:
        r1, f1 = cap_best.read()
        r2, f2 = cap_crime.read()
        if not r1 and not r2:
            break
        if r1:
            f1 = cv2.resize(f1, (target_w_single, target_h_single))
            f1 = enhance_lowlight(f1)
            landmarks1, _ = pose.extract_landmarks(f1)
            f1 = draw_force_lines(f1, landmarks1, target_h, w1n)
            cv2.putText(f1, f"SUSPECT {best['id']} (BEST MATCH)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(f1, f"Match: {best['avg_prob']*100:.1f}%", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
        if r2:
            f2 = cv2.resize(f2, (target_w_single, target_h_single))
            f2 = enhance_lowlight(f2)
            landmarks2, _ = pose.extract_landmarks(f2)
            f2 = draw_force_lines(f2, landmarks2, target_h, w2n)
            cv2.putText(f2, "CRIME SCENE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        out.write(np.hstack([f1, f2]) if r1 and r2 else (f1 if r1 else f2))
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")
    
    cap_best.release()
    cap_crime.release()
    out.release()
    print(f"✅ 对比视频（带骨架）: output/comparison_with_skeleton.mp4")
