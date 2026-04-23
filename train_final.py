import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
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

print("="*60)
print("最终版训练（随机森林 + 改进特征）")
print("="*60)

pose = PoseAnalyzer(use_yolo=False)
gait = GaitAnalyzer()

video_features = []
for f in os.listdir("videos"):
    if f.endswith(".mp4"):
        print(f"处理: {f}")
        feat = extract_features_improved(f"videos/{f}", pose, gait)
        if feat is not None:
            video_features.append({"file": f, "features": feat})
            print(f"  ✅ 特征维度: {len(feat)}")
        else:
            print(f"  ❌ 失败")

print(f"\n成功提取 {len(video_features)} 个视频")

# 分组
person_groups = {
    "person_A": [],
    "person_B": [],
    "person_C": [],
}

for v in video_features:
    file = v["file"]
    if file == "crime_scene.mp4":
        person_groups["person_A"].append(v)
    elif file.startswith("suspect_001"):
        person_groups["person_A"].append(v)
    elif file.startswith("suspect_002"):
        person_groups["person_B"].append(v)
    elif file.startswith("suspect_003"):
        person_groups["person_C"].append(v)

print("\n分组结果：")
for person, videos in person_groups.items():
    print(f"  {person}: {len(videos)} 个视频")

# 训练对
X, y = [], []
for person, videos in person_groups.items():
    if len(videos) >= 2:
        for i in range(len(videos)):
            for j in range(i+1, len(videos)):
                diff = np.abs(videos[i]["features"] - videos[j]["features"])
                prod = videos[i]["features"] * videos[j]["features"]
                pair = np.concatenate([diff, prod, videos[i]["features"], videos[j]["features"]])
                X.append(pair)
                y.append(1)

persons = list(person_groups.keys())
for i in range(len(persons)):
    for j in range(i+1, len(persons)):
        if person_groups[persons[i]] and person_groups[persons[j]]:
            f1 = person_groups[persons[i]][0]["features"]
            f2 = person_groups[persons[j]][0]["features"]
            diff = np.abs(f1 - f2)
            prod = f1 * f2
            pair = np.concatenate([diff, prod, f1, f2])
            X.append(pair)
            y.append(0)

X, y = np.array(X), np.array(y)
print(f"\n总样本对: {len(X)} (正: {sum(y)}, 负: {len(y)-sum(y)})")

if len(X) < 10:
    print("样本不足")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ 准确率: {acc*100:.1f}%")

joblib.dump(model, "gait_model_final.pkl")
print("✅ 模型已保存: gait_model_final.pkl")
