import cv2
import os
from pose import PoseAnalyzer
from enhance import enhance_lowlight

def process_video_with_yolo(input_path, output_path):
    """用YOLOv5检测人体，生成跟踪视频"""
    
    # 初始化 YOLOv5（use_yolo=True）
    pose = PoseAnalyzer(use_yolo=True)
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 输出视频（保持原尺寸）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print(f"处理: {os.path.basename(input_path)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 可选：增强（低照度时开启）
        # frame = enhance_lowlight(frame)
        
        # YOLOv5 检测人体
        bbox = pose.detect_person(frame)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            # 画绿色框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 加标签
            cv2.putText(frame, "Person", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示帧数（可选）
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  已处理 {frame_count} 帧")
    
    cap.release()
    out.release()
    print(f"✅ 完成: {output_path}\n")


def main():
    print("="*60)
    print("YOLOv5 人体检测跟踪 - 生成三个嫌疑人跟踪视频")
    print("="*60)
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    # 三个嫌疑人的 normal 视频
    suspects = [
        ("videos/suspect_001_normal.mp4", "output/track_suspect_001.mp4"),
        ("videos/suspect_002_normal.mp4", "output/track_suspect_002.mp4"),
        ("videos/suspect_003_normal.mp4", "output/track_suspect_003.mp4"),
    ]
    
    for input_path, output_path in suspects:
        if os.path.exists(input_path):
            process_video_with_yolo(input_path, output_path)
        else:
            print(f"❌ 文件不存在: {input_path}")
    
    print("="*60)
    print("全部完成！输出文件在 output/ 目录")
    print("  - track_suspect_001.mp4")
    print("  - track_suspect_002.mp4")
    print("  - track_suspect_003.mp4")
    print("="*60)


if __name__ == "__main__":
    main()
