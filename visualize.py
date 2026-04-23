import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_analysis_overlay(image, results):
    """基础版可视化（字幕已缩小）"""
    overlay = image.copy()

    if results.get('symmetry_score'):
        score = results['symmetry_score']
        if score >= 85:
            color = (0, 255, 0)
        elif score >= 70:
            color = (0, 255, 255)
        elif score >= 50:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        text = f"Symmetry: {score:.1f}"
        cv2.putText(overlay, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if results.get('avg_step_length'):
        step = results['avg_step_length']
        text = f"Step: {step:.2f}"
        cv2.putText(overlay, text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if results.get('recommendation'):
        text = results['recommendation']
        cv2.putText(overlay, text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return overlay


def draw_advanced_overlay(image, results, pose_landmarks=None):
    """增强版可视化 - 字幕移到角落，不遮挡人体"""
    h, w = image.shape[:2]
    overlay = image.copy()

    # 1. 绘制骨架
    if pose_landmarks:
        import mediapipe as mp
        mp_draw = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_draw.draw_landmarks(
            overlay,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    overlay_alpha = cv2.addWeighted(overlay, 0.8, image, 0.2, 0)

    # 2. 对称性评分仪表盘（右上角，缩小）
    if results.get('symmetry_score'):
        score = results['symmetry_score']

        if score >= 85:
            color = (0, 255, 0)
        elif score >= 70:
            color = (0, 255, 255)
        elif score >= 50:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        # 仪表盘移到右上角，缩小尺寸
        cv2.circle(overlay_alpha, (w - 50, 50), 30, (50, 50, 50), -1)
        cv2.circle(overlay_alpha, (w - 50, 50), 30, color, 2)
        cv2.putText(overlay_alpha, f'{score:.0f}', (w - 68, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(overlay_alpha, 'Sym', (w - 60, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 3. 关节角度面板（右上角，仪表盘下方）
    if results.get('left_knee_avg') and results.get('right_knee_avg'):
        panel_x = w - 105
        panel_y = 95
        cv2.rectangle(overlay_alpha, (panel_x, panel_y), (w - 10, panel_y + 50),
                      (0, 0, 0), -1)
        cv2.rectangle(overlay_alpha, (panel_x, panel_y), (w - 10, panel_y + 50),
                      (100, 100, 100), 1)

        left_knee = results['left_knee_avg']
        right_knee = results['right_knee_avg']

        cv2.putText(overlay_alpha, f'L:{left_knee:.0f}',
                    (panel_x + 5, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 100), 1)
        cv2.putText(overlay_alpha, f'R:{right_knee:.0f}',
                    (panel_x + 5, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)

    # 4. 建议（左上角）
    if results.get('recommendation'):
        text = results['recommendation']
        cv2.putText(overlay_alpha, text, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return overlay_alpha


def draw_occlusion_warning(image, results):
    """遮挡警告（左上角）"""
    if not results:
        return image

    overlay = image.copy()
    occlusion = results.get('occlusion')
    reliability = results.get('reliability', 100)

    if not occlusion:
        return overlay

    ratio = occlusion.get('occluded_frame_ratio', 0)

    if ratio > 0.5:
        color = (0, 0, 255)
        status = "HIGH OCCLUSION"
    elif ratio > 0.2:
        color = (0, 165, 255)
        status = "MODERATE OCCLUSION"
    elif ratio > 0.05:
        color = (0, 255, 255)
        status = "LIGHT OCCLUSION"
    else:
        color = (0, 255, 0)
        status = "CLEAR"

    cv2.putText(overlay, f"Occlusion: {status} ({ratio:.0%})",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if reliability < 50:
        rel_color = (0, 0, 255)
    elif reliability < 80:
        rel_color = (0, 165, 255)
    else:
        rel_color = (0, 255, 0)

    cv2.putText(overlay, f"Reliability: {reliability:.0f}%",
                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.35, rel_color, 1)

    return overlay


def create_symmetry_heatmap(keypoints_history, frame_shape):
    """左右对称性热力图"""
    left_heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
    right_heatmap = np.zeros(frame_shape[:2], dtype=np.float32)

    for keypoints in keypoints_history:
        for name, (x, y, vis) in keypoints.items():
            if vis > 0.5:
                h, w = frame_shape[:2]
                px, py = int(x * w), int(y * h)
                if 0 <= px < w and 0 <= py < h:
                    if 'left' in name:
                        left_heatmap[py, px] += 1
                    elif 'right' in name:
                        right_heatmap[py, px] += 1

    if np.max(left_heatmap) > 0:
        left_heatmap = left_heatmap / np.max(left_heatmap)
    if np.max(right_heatmap) > 0:
        right_heatmap = right_heatmap / np.max(right_heatmap)

    symmetry_img = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
    symmetry_img[:, :, 2] = (left_heatmap * 255).astype(np.uint8)
    symmetry_img[:, :, 0] = (right_heatmap * 255).astype(np.uint8)

    cv2.putText(symmetry_img, 'Left Leg (Red)', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(symmetry_img, 'Right Leg (Blue)', (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return symmetry_img


def create_gait_energy_image(keypoints_history, frame_shape):
    """步态能量图"""
    heatmap = np.zeros(frame_shape[:2], dtype=np.float32)

    for keypoints in keypoints_history:
        for name, (x, y, vis) in keypoints.items():
            if vis > 0.5:
                h, w = frame_shape[:2]
                px, py = int(x * w), int(y * h)
                if 0 <= px < w and 0 <= py < h:
                    heatmap[py, px] += 1

    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.putText(heatmap_colored, 'Gait Energy Image',
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return heatmap_colored


def create_occlusion_heatmap(occlusion_history, frame_shape):
    """遮挡热力图"""
    if not occlusion_history:
        return np.zeros(frame_shape[:2], dtype=np.uint8)

    joint_to_region = {
        'left_shoulder': (0.4, 0.25), 'right_shoulder': (0.6, 0.25),
        'left_elbow': (0.35, 0.35), 'right_elbow': (0.65, 0.35),
        'left_wrist': (0.3, 0.4), 'right_wrist': (0.7, 0.4),
        'left_hip': (0.4, 0.55), 'right_hip': (0.6, 0.55),
        'left_knee': (0.4, 0.75), 'right_knee': (0.6, 0.75),
        'left_ankle': (0.4, 0.9), 'right_ankle': (0.6, 0.9),
    }

    heatmap = np.zeros(frame_shape[:2], dtype=np.float32)

    for occ in occlusion_history:
        for point in occ['points']:
            if point in joint_to_region:
                x, y = joint_to_region[point]
                px, py = int(x * frame_shape[1]), int(y * frame_shape[0])
                if 0 <= px < frame_shape[1] and 0 <= py < frame_shape[0]:
                    heatmap[py, px] += 1

    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    cv2.putText(heatmap_colored, 'Occlusion Heatmap',
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return heatmap_colored


def plot_knee_angle_curves(gait_analyzer, output_path=None):
    """绘制膝关节角度曲线"""
    if len(gait_analyzer.keypoints_history) < 10:
        return None

    left_angles = []
    right_angles = []

    for kp in gait_analyzer.keypoints_history:
        left = gait_analyzer.calculate_knee_angle(kp, 'left')
        right = gait_analyzer.calculate_knee_angle(kp, 'right')
        if left:
            left_angles.append(left)
        if right:
            right_angles.append(right)

    min_len = min(len(left_angles), len(right_angles))
    left_angles = left_angles[:min_len]
    right_angles = right_angles[:min_len]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(left_angles, 'r-', label='Left Knee', linewidth=2)
    plt.plot(right_angles, 'b-', label='Right Knee', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('Angle (deg)')
    plt.title('Knee Angle Variation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 180)

    plt.subplot(1, 2, 2)
    avg_left = np.mean(left_angles) if left_angles else 0
    avg_right = np.mean(right_angles) if right_angles else 0

    plt.bar(['Left Knee', 'Right Knee'], [avg_left, avg_right],
            color=['red', 'blue'], alpha=0.7)
    plt.ylabel('Average Angle (deg)')
    plt.title('Left vs Right Knee Angle')
    plt.ylim(0, 180)

    plt.text(0, avg_left + 3, f'{avg_left:.1f}', ha='center', fontsize=10)
    plt.text(1, avg_right + 3, f'{avg_right:.1f}', ha='center', fontsize=10)

    diff = abs(avg_left - avg_right)
    plt.text(0.5, max(avg_left, avg_right) + 15,
             f'Diff: {diff:.1f} deg', ha='center', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return plt.gcf()


def create_symmetry_dashboard(symmetry_score, recommendation, output_path=None):
    """创建对称性仪表盘"""
    fig, ax = plt.subplots(figsize=(5, 4))

    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    thresholds = [0, 20, 40, 60, 80, 100]

    for i in range(len(thresholds) - 1):
        start = thresholds[i] / 100 * np.pi
        end = thresholds[i + 1] / 100 * np.pi
        theta_seg = np.linspace(start, end, 50)
        x = np.cos(theta_seg)
        y = np.sin(theta_seg)
        ax.fill_between(x, 0, y, color=colors[i], alpha=0.5)

    angle = symmetry_score / 100 * np.pi
    ax.arrow(0, 0, 0.7 * np.cos(angle), 0.7 * np.sin(angle),
             head_width=0.08, head_length=0.08, fc='black', ec='black', linewidth=2)

    if symmetry_score >= 85:
        score_color = 'green'
    elif symmetry_score >= 70:
        score_color = 'orange'
    else:
        score_color = 'red'

    ax.text(0, -0.25, f'{symmetry_score:.1f}', fontsize=32,
            ha='center', fontweight='bold', color=score_color)
    ax.text(0, -0.4, 'Symmetry Score', fontsize=12, ha='center')

    rec_text = recommendation if recommendation else "Insufficient data"
    ax.text(0, -0.55, rec_text, fontsize=9, ha='center',
            color='green' if symmetry_score >= 70 else 'orange')

    for val, label in [(0, '0'), (25, '25'), (50, '50'), (75, '75'), (100, '100')]:
        angle_rad = val / 100 * np.pi
        x = 1.05 * np.cos(angle_rad)
        y = 1.05 * np.sin(angle_rad)
        ax.text(x, y, label, ha='center', va='center', fontsize=8)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Gait Symmetry', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig