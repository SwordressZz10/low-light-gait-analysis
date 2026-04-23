import cv2
import numpy as np


def enhance_lowlight(image, method='progressive'):
    """
    低照度图像增强
    method: 'progressive' - 渐进式增强
            'histogram' - 直方图均衡
    """
    if method == 'progressive':
        return progressive_enhance(image)
    elif method == 'histogram':
        return histogram_equalize(image)
    else:
        return image


def progressive_enhance(image, stages=3):

    # 转为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 明度通道增强
    v_enhanced = v.astype(np.float32)

    for stage in range(stages):
        # 暗区掩码（亮度<128）
        dark_mask = (v_enhanced < 128)

        # 渐进Gamma校正
        gamma = 0.7 + stage * 0.15
        v_enhanced[dark_mask] = 255 * ((v_enhanced[dark_mask] / 255) ** gamma)

        # 最后阶段做对比度拉伸
        if stage == stages - 1:
            p5 = np.percentile(v_enhanced, 5)
            p95 = np.percentile(v_enhanced, 95)
            v_enhanced = np.clip((v_enhanced - p5) / (p95 - p5) * 255, 0, 255)

    v_enhanced = v_enhanced.astype(np.uint8)

    # 合并并转回BGR
    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return enhanced


def histogram_equalize(image):
    """直方图均衡化"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)