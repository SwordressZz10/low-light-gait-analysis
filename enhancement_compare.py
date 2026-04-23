import cv2
import numpy as np
import os
from enhance import enhance_lowlight

print("="*60)
print("生成犯罪现场增强前后对比视频")
print("="*60)

crime_file = "videos/crime_scene.mp4"

if not os.path.exists(crime_file):
    print(f"❌ 找不到文件: {crime_file}")
    exit()

cap = cv2.VideoCapture(crime_file)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 统一高度为 480
target_h = 480
new_w = int(w * target_h / h)

os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/enhancement_comparison.mp4", fourcc, fps, (new_w * 2, target_h))

frame_count = 0

print("处理中...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 原图
    original = cv2.resize(frame, (new_w, target_h))
    
    # 增强后
    enhanced = enhance_lowlight(frame)
    enhanced = cv2.resize(enhanced, (new_w, target_h))
    
    # 添加文字标签
    cv2.putText(original, "ORIGINAL (Low Light)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(enhanced, "ENHANCED (Progressive Enhancement)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 并排显示
    comparison = np.hstack([original, enhanced])
    out.write(comparison)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"已处理 {frame_count} 帧")

cap.release()
out.release()

print(f"\n✅ 增强对比视频已生成: output/enhancement_comparison.mp4")
print("   - 左侧: 原始低照度画面")
print("   - 右侧: 渐进式增强后画面")
