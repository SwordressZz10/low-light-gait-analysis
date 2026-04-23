import cv2
import numpy as np
from pose import PoseAnalyzer

def detect_all_persons(image, pose, iou_threshold=0.3):
    """重复检测直到找到所有人"""
    h, w = image.shape[:2]
    all_bboxes = []
    current_image = image.copy()
    
    while True:
        # 检测一个人
        bbox = pose.detect_person(current_image)
        
        if bbox is None:
            break
        
        x1, y1, x2, y2 = bbox
        
        # 检查是否和已有框重叠（避免重复检测同一人）
        overlap = False
        for (ex1, ey1, ex2, ey2) in all_bboxes:
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)
            if ix2 > ix1 and iy2 > iy1:
                overlap_area = (ix2 - ix1) * (iy2 - iy1)
                bbox_area = (x2 - x1) * (y2 - y1)
                if overlap_area / bbox_area > iou_threshold:
                    overlap = True
                    break
        
        if not overlap:
            all_bboxes.append((x1, y1, x2, y2))
        
        # 屏蔽已检测区域
        cv2.rectangle(current_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
    return all_bboxes


# 主程序
img = cv2.imread('videos/test_new.jpg')
if img is None:
    print('❌ 无法读取图片')
    exit()

print(f'✅ 图片尺寸: {img.shape}')

pose = PoseAnalyzer(use_yolo=True)
bboxes = detect_all_persons(img, pose)

if bboxes:
    print(f'✅ 检测到 {len(bboxes)} 个人')
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f'Person {i+1}', (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f'   Person {i+1}: ({x1},{y1}) → ({x2},{y2})')
    
    cv2.imwrite('output/detection_result.jpg', img)
    print('✅ 保存至: output/detection_result.jpg')
else:
    print('❌ 未检测到人体')
