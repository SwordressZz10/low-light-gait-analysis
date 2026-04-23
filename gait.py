import numpy as np
from pose import calculate_angle
from collections import Counter


class GaitAnalyzer:
    def __init__(self):
        self.keypoints_history = []
        self.occlusion_history = []
        self.visibility_threshold = 0.3

    def add_frame(self, keypoints):
        if keypoints:
            self.keypoints_history.append(keypoints)
            is_occluded, ratio, points = self.detect_occlusion(keypoints)
            self.occlusion_history.append({
                'is_occluded': is_occluded,
                'ratio': ratio,
                'points': points
            })

    def detect_occlusion(self, keypoints, threshold=0.3):
        if not keypoints:
            return True, 1.0, []

        occluded_points = []
        for name, (x, y, vis) in keypoints.items():
            if vis < threshold:
                occluded_points.append(name)

        occlusion_ratio = len(occluded_points) / len(keypoints)
        is_occluded = occlusion_ratio > 0.2

        return is_occluded, occlusion_ratio, occluded_points

    def calculate_step_length(self, keypoints):
        left_ankle = keypoints.get('left_ankle')
        right_ankle = keypoints.get('right_ankle')

        if left_ankle and right_ankle:
            dx = left_ankle[0] - right_ankle[0]
            dy = left_ankle[1] - right_ankle[1]
            return np.sqrt(dx * dx + dy * dy)
        return None

    def calculate_knee_angle(self, keypoints, side='left'):
        if side == 'left':
            hip = keypoints.get('left_hip')
            knee = keypoints.get('left_knee')
            ankle = keypoints.get('left_ankle')
        else:
            hip = keypoints.get('right_hip')
            knee = keypoints.get('right_knee')
            ankle = keypoints.get('right_ankle')

        if hip and knee and ankle:
            return calculate_angle(hip, knee, ankle)
        return None

    def infer_missing_joints(self, keypoints):
        inferred = keypoints.copy()

        if 'left_knee' not in keypoints or keypoints.get('left_knee')[2] < 0.3:
            if 'right_knee' in keypoints and keypoints['right_knee'][2] > 0.5:
                right_knee = keypoints['right_knee']
                if 'left_hip' in inferred:
                    inferred['left_knee'] = (
                        2 * inferred['left_hip'][0] - right_knee[0],
                        right_knee[1],
                        0.5
                    )

        if 'left_ankle' not in keypoints or keypoints.get('left_ankle')[2] < 0.3:
            if 'right_ankle' in keypoints and keypoints['right_ankle'][2] > 0.5:
                right_ankle = keypoints['right_ankle']
                if 'left_hip' in inferred:
                    inferred['left_ankle'] = (
                        2 * inferred['left_hip'][0] - right_ankle[0],
                        right_ankle[1],
                        0.5
                    )

        return inferred

    def get_occlusion_report(self):
        if not self.occlusion_history:
            return None

        occluded_frames = sum(1 for o in self.occlusion_history if o['is_occluded'])
        total_frames = len(self.occlusion_history)

        all_points = []
        for o in self.occlusion_history:
            all_points.extend(o['points'])

        most_occluded = Counter(all_points).most_common(5)

        return {
            'occluded_frame_ratio': occluded_frames / total_frames if total_frames > 0 else 0,
            'avg_occlusion_ratio': np.mean([o['ratio'] for o in self.occlusion_history]),
            'most_occluded_joints': most_occluded,
            'total_frames': total_frames,
            'occluded_frames': occluded_frames
        }

    def get_reliability_score(self, occlusion_report):
        if not occlusion_report:
            return 100

        occlusion_ratio = occlusion_report.get('occluded_frame_ratio', 0)

        if occlusion_ratio == 0:
            return 100
        elif occlusion_ratio < 0.1:
            return 90
        elif occlusion_ratio < 0.2:
            return 75
        elif occlusion_ratio < 0.3:
            return 60
        elif occlusion_ratio < 0.5:
            return 40
        else:
            return 20

    def analyze(self):
        if len(self.keypoints_history) < 10:
            return {"error": "Insufficient data, need at least 10 frames"}

        occlusion_report = self.get_occlusion_report()

        step_lengths = []
        left_knee_angles = []
        right_knee_angles = []

        for kp in self.keypoints_history:
            kp_inferred = self.infer_missing_joints(kp)

            step = self.calculate_step_length(kp_inferred)
            if step:
                step_lengths.append(step)

            left_angle = self.calculate_knee_angle(kp_inferred, 'left')
            right_angle = self.calculate_knee_angle(kp_inferred, 'right')
            if left_angle:
                left_knee_angles.append(left_angle)
            if right_angle:
                right_knee_angles.append(right_angle)

        results = {
            'avg_step_length': np.mean(step_lengths) if step_lengths else None,
            'step_length_std': np.std(step_lengths) if step_lengths else None,
            'left_knee_avg': np.mean(left_knee_angles) if left_knee_angles else None,
            'right_knee_avg': np.mean(right_knee_angles) if right_knee_angles else None,
        }

        if results['left_knee_avg'] and results['right_knee_avg']:
            knee_diff = abs(results['left_knee_avg'] - results['right_knee_avg'])
            results['symmetry_score'] = max(0, 100 - knee_diff * 2)
        else:
            results['symmetry_score'] = None

        results['recommendation'] = self.get_recommendation(results)

        if occlusion_report:
            results['occlusion'] = occlusion_report
            results['reliability'] = self.get_reliability_score(occlusion_report)

        return results

    def get_recommendation(self, results):
        if results['symmetry_score'] is None:
            return "Insufficient data"

        score = results['symmetry_score']
        if score >= 85:
            return "Good, keep it up"
        elif score >= 70:
            return "Mild asymmetry, focus on weak side"
        elif score >= 50:
            return "Moderate asymmetry, consult therapist"
        else:
            return "Severe asymmetry, seek medical help"

    def clear(self):
        self.keypoints_history = []
        self.occlusion_history = []