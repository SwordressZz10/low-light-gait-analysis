import cv2
from pose import PoseAnalyzer

def detect_and_save(image_path, output_path):
    """检测图片中的人体，画框并保存"""
    
    # 初始化 YOLOv5
    pose = PoseAnalyzer(use_yolo=True)
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return False
    
    # 检测人体
    bbox = pose.detect_person(img)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        # 画绿色框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # 加标签
        cv2.putText(img, "Person", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"✅ 检测到人体: ({x1},{y1}) → ({x2},{y2})")
    else:
        print("❌ 未检测到人体")
    
    # 保存结果
    cv2.imwrite(output_path, img)
    print(f"✅ 已保存: {output_path}")
    return True


if __name__ == "__main__":
    # 使用 videos 文件夹中的 image1.jpg
    detect_and_save("videos/image1.jpg", "output/detection_result.jpg")
