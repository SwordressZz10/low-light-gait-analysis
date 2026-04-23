# 低照度步态分析系统

基于渐进式图像增强、YOLOv5 和 MediaPipe 的低照度步态分析系统。

## 主要特性
- 渐进式图像增强（HSV空间三阶段Gamma校正）
- YOLOv5 人体检测
- MediaPipe 姿态估计（33个关键点，使用12个）
- 步态参数计算（膝关节角度、步长、对称性评分）
- 遮挡检测与关节推断
- 随机森林匹配（58维特征，准确率100%）
- 丰富可视化（能量图、热力图、仪表盘、角度曲线、对比视频）

## 项目结构
- enhance.py: 图像增强
- pose.py: YOLOv5+MediaPipe融合
- gait.py: 步态分析与遮挡检测
- visualize.py: 可视化
- train_final.py: 训练模型
- match_final.py: 嫌疑人匹配
- match_with_skeleton.py: 带骨架的匹配

## 实验数据
- CASIA Dataset C（153人红外数据）
- 自采视频（犯罪现场+3名嫌疑人）

## 结果
- 模型测试准确率：100%
- 增强后关键点检测率提升：约15%
- 最佳嫌疑人匹配：87.7%

## 环境
Python 3.10, opencv-python, mediapipe, torch, scikit-learn, joblib, numpy, matplotlib

## 致谢
符长虹老师（同济大学），中科院自动化所CASIA数据集
