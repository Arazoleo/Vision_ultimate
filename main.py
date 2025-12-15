import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import time
import math
import random

COLORS = {
    'neon_green': (0, 255, 128),
    'neon_pink': (255, 0, 255),
    'neon_blue': (255, 191, 0),
    'neon_yellow': (0, 255, 255),
    'neon_red': (0, 0, 255),
    'neon_cyan': (255, 255, 0),
    'neon_orange': (0, 165, 255),
    'neon_purple': (255, 0, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'dark_gray': (40, 40, 40),
}

EMOTIONS = {
    'happy': {'color': COLORS['neon_green'], 'label': 'Feliz'},
    'sad': {'color': COLORS['neon_blue'], 'label': 'Triste'},
    'angry': {'color': COLORS['neon_red'], 'label': 'Raiva'},
    'surprise': {'color': COLORS['neon_yellow'], 'label': 'Surpreso'},
    'fear': {'color': COLORS['neon_purple'], 'label': 'Medo'},
    'disgust': {'color': COLORS['neon_orange'], 'label': 'Nojo'},
    'neutral': {'color': COLORS['gray'], 'label': 'Neutro'},
}

MODES = ['DETECTION', 'EMOTION', 'CANVAS', 'EFFECTS', 'FOCUS', 'FITNESS']

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class EmotionDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.emotion_history = deque(maxlen=10)
        self.current_emotion = 'neutral'
        
        self.calibration_data = {
            'mouth_curve': deque(maxlen=60),
            'eye_openness': deque(maxlen=60),
            'brow_height': deque(maxlen=60),
            'brow_distance': deque(maxlen=60)
        }
        self.baseline = {}
        self.calibrated = False
        
        self.UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.LIP_TOP_CENTER = 0
        self.LIP_BOTTOM_CENTER = 17
        self.LIP_TOP_INNER = 13
        self.LIP_BOTTOM_INNER = 14
        
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.LEFT_EYE_LEFT = 33
        self.LEFT_EYE_RIGHT = 133
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374
        self.RIGHT_EYE_LEFT = 362
        self.RIGHT_EYE_RIGHT = 263
        
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107]
        self.RIGHT_EYEBROW = [300, 293, 334, 296, 336]
        self.LEFT_EYEBROW_INNER = 107
        self.LEFT_EYEBROW_OUTER = 70
        self.RIGHT_EYEBROW_INNER = 336
        self.RIGHT_EYEBROW_OUTER = 300
        
        self.NOSE_TIP = 4
        self.FOREHEAD = 10
        
        self.debug_info = {}
    
    def _get_landmark_pos(self, landmarks, idx, w, h):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
    
    def _calculate_eye_aspect_ratio(self, landmarks):
        left_height = abs(landmarks[self.LEFT_EYE_TOP].y - landmarks[self.LEFT_EYE_BOTTOM].y)
        left_width = abs(landmarks[self.LEFT_EYE_LEFT].x - landmarks[self.LEFT_EYE_RIGHT].x)
        right_height = abs(landmarks[self.RIGHT_EYE_TOP].y - landmarks[self.RIGHT_EYE_BOTTOM].y)
        right_width = abs(landmarks[self.RIGHT_EYE_LEFT].x - landmarks[self.RIGHT_EYE_RIGHT].x)
        
        left_ear = left_height / (left_width + 0.001)
        right_ear = right_height / (right_width + 0.001)
        
        return (left_ear + right_ear) / 2
    
    def _calculate_eyebrow_height(self, landmarks):
        left_brow_y = np.mean([landmarks[i].y for i in self.LEFT_EYEBROW])
        right_brow_y = np.mean([landmarks[i].y for i in self.RIGHT_EYEBROW])
        left_eye_y = landmarks[self.LEFT_EYE_TOP].y
        right_eye_y = landmarks[self.RIGHT_EYE_TOP].y
        
        left_dist = left_eye_y - left_brow_y
        right_dist = right_eye_y - right_brow_y
        
        return (left_dist + right_dist) / 2 * 1000
    
    def _calculate_eyebrow_distance(self, landmarks):
        left_inner = landmarks[self.LEFT_EYEBROW_INNER]
        right_inner = landmarks[self.RIGHT_EYEBROW_INNER]
        
        distance = abs(right_inner.x - left_inner.x)
        return distance * 1000
    
    def _calculate_eyebrow_angle(self, landmarks):
        left_inner = landmarks[self.LEFT_EYEBROW_INNER]
        left_outer = landmarks[self.LEFT_EYEBROW_OUTER]
        right_inner = landmarks[self.RIGHT_EYEBROW_INNER]
        right_outer = landmarks[self.RIGHT_EYEBROW_OUTER]
        
        left_angle = (left_inner.y - left_outer.y) * 1000
        right_angle = (right_inner.y - right_outer.y) * 1000
        
        return (left_angle + right_angle) / 2
    
    def _calculate_mouth_curve(self, landmarks):
        left_corner = landmarks[self.MOUTH_LEFT]
        right_corner = landmarks[self.MOUTH_RIGHT]
        top_center = landmarks[self.LIP_TOP_CENTER]
        bottom_center = landmarks[self.LIP_BOTTOM_CENTER]
        mouth_center_y = (top_center.y + bottom_center.y) / 2
        corners_y = (left_corner.y + right_corner.y) / 2
        curve_score = (mouth_center_y - corners_y) * 1000
        return curve_score
    
    def _calculate_mouth_openness(self, landmarks):
        top = landmarks[self.LIP_TOP_INNER]
        bottom = landmarks[self.LIP_BOTTOM_INNER]
        return abs(bottom.y - top.y) * 1000
    
    def _calculate_mouth_width(self, landmarks):
        left = landmarks[self.MOUTH_LEFT]
        right = landmarks[self.MOUTH_RIGHT]
        return abs(right.x - left.x) * 1000
    
    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return self.current_emotion, 0.5, None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        eye_ar = self._calculate_eye_aspect_ratio(landmarks)
        brow_height = self._calculate_eyebrow_height(landmarks)
        brow_distance = self._calculate_eyebrow_distance(landmarks)
        brow_angle = self._calculate_eyebrow_angle(landmarks)
        mouth_curve = self._calculate_mouth_curve(landmarks)
        mouth_open = self._calculate_mouth_openness(landmarks)
        mouth_width = self._calculate_mouth_width(landmarks)
        
        self.calibration_data['mouth_curve'].append(mouth_curve)
        self.calibration_data['eye_openness'].append(eye_ar)
        self.calibration_data['brow_height'].append(brow_height)
        self.calibration_data['brow_distance'].append(brow_distance)
        
        if len(self.calibration_data['mouth_curve']) >= 30 and not self.calibrated:
            for key in self.calibration_data:
                sorted_vals = sorted(self.calibration_data[key])
                self.baseline[key] = sorted_vals[len(sorted_vals) // 2]
            self.calibrated = True
        
        if not self.calibrated:
            return 'neutral', 0.5, None
        
        norm_mouth = mouth_curve - self.baseline['mouth_curve']
        norm_eyes = (eye_ar - self.baseline['eye_openness']) * 100
        norm_brow_h = brow_height - self.baseline['brow_height']
        norm_brow_d = brow_distance - self.baseline['brow_distance']
        
        self.debug_info = {
            'mouth_curve': norm_mouth,
            'eye_open': norm_eyes,
            'brow_height': norm_brow_h,
            'brow_distance': norm_brow_d,
            'brow_angle': brow_angle,
            'mouth_open': mouth_open,
            'mouth_width': mouth_width
        }
        
        scores = {
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'surprise': 0,
            'fear': 0,
            'neutral': 50
        }
        
        if norm_mouth > 2:
            scores['happy'] += 30 + norm_mouth * 5
        if norm_mouth > 1 and mouth_width > self.baseline.get('mouth_width', 50):
            scores['happy'] += 20
        if norm_eyes < -0.5:
            scores['happy'] += 15
        
        if norm_eyes > 1.5:
            scores['surprise'] += 40
        if norm_brow_h > 3:
            scores['surprise'] += 30
        if mouth_open > 30:
            scores['surprise'] += 30
        
        if norm_brow_d < -5:
            scores['angry'] += 40
        if brow_angle > 2:
            scores['angry'] += 25
        if norm_eyes < -0.3:
            scores['angry'] += 20
        if norm_mouth < -1:
            scores['angry'] += 15
        
        if norm_mouth < -2:
            scores['sad'] += 30
        if norm_brow_h > 2 and brow_angle < -1:
            scores['sad'] += 25
        if norm_eyes < 0:
            scores['sad'] += 15
        
        if norm_eyes > 1:
            scores['fear'] += 25
        if norm_brow_h > 2:
            scores['fear'] += 20
        if norm_brow_d < -3:
            scores['fear'] += 15
        if mouth_open > 15 and mouth_open < 30:
            scores['fear'] += 15
        
        max_emotion = max(scores, key=scores.get)
        max_score = scores[max_emotion]
        
        if max_score < 40:
            max_emotion = 'neutral'
            confidence = 0.7
        else:
            confidence = min(0.95, max_score / 100)
        
        self.debug_info['scores'] = scores
        
        self.emotion_history.append(max_emotion)
        
        if len(self.emotion_history) >= 3:
            emotion_counts = {}
            for i, e in enumerate(self.emotion_history):
                weight = 1 + (i * 0.3)
                emotion_counts[e] = emotion_counts.get(e, 0) + weight
            self.current_emotion = max(emotion_counts, key=emotion_counts.get)
        else:
            self.current_emotion = max_emotion
        
        return self.current_emotion, confidence, landmarks
    
    def draw_face_debug(self, frame, landmarks):
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        for i in range(len(self.UPPER_LIP_OUTER) - 1):
            pt1 = self._get_landmark_pos(landmarks, self.UPPER_LIP_OUTER[i], w, h)
            pt2 = self._get_landmark_pos(landmarks, self.UPPER_LIP_OUTER[i + 1], w, h)
            cv2.line(frame, pt1, pt2, COLORS['neon_pink'], 2)
        
        for i in range(len(self.LOWER_LIP_OUTER) - 1):
            pt1 = self._get_landmark_pos(landmarks, self.LOWER_LIP_OUTER[i], w, h)
            pt2 = self._get_landmark_pos(landmarks, self.LOWER_LIP_OUTER[i + 1], w, h)
            cv2.line(frame, pt1, pt2, COLORS['neon_cyan'], 2)
        
        for i in range(len(self.LEFT_EYEBROW) - 1):
            pt1 = self._get_landmark_pos(landmarks, self.LEFT_EYEBROW[i], w, h)
            pt2 = self._get_landmark_pos(landmarks, self.LEFT_EYEBROW[i + 1], w, h)
            cv2.line(frame, pt1, pt2, COLORS['neon_yellow'], 2)
        
        for i in range(len(self.RIGHT_EYEBROW) - 1):
            pt1 = self._get_landmark_pos(landmarks, self.RIGHT_EYEBROW[i], w, h)
            pt2 = self._get_landmark_pos(landmarks, self.RIGHT_EYEBROW[i + 1], w, h)
            cv2.line(frame, pt1, pt2, COLORS['neon_yellow'], 2)
        
        left_eye_pts = [self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, self.LEFT_EYE_LEFT, self.LEFT_EYE_RIGHT]
        right_eye_pts = [self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM, self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT]
        
        for idx in left_eye_pts:
            pt = self._get_landmark_pos(landmarks, idx, w, h)
            cv2.circle(frame, pt, 3, COLORS['neon_green'], -1)
        
        for idx in right_eye_pts:
            pt = self._get_landmark_pos(landmarks, idx, w, h)
            cv2.circle(frame, pt, 3, COLORS['neon_green'], -1)
        
        left_eye_top = self._get_landmark_pos(landmarks, self.LEFT_EYE_TOP, w, h)
        left_eye_bottom = self._get_landmark_pos(landmarks, self.LEFT_EYE_BOTTOM, w, h)
        right_eye_top = self._get_landmark_pos(landmarks, self.RIGHT_EYE_TOP, w, h)
        right_eye_bottom = self._get_landmark_pos(landmarks, self.RIGHT_EYE_BOTTOM, w, h)
        
        cv2.line(frame, left_eye_top, left_eye_bottom, COLORS['neon_orange'], 1)
        cv2.line(frame, right_eye_top, right_eye_bottom, COLORS['neon_orange'], 1)
        
        left_brow_inner = self._get_landmark_pos(landmarks, self.LEFT_EYEBROW_INNER, w, h)
        right_brow_inner = self._get_landmark_pos(landmarks, self.RIGHT_EYEBROW_INNER, w, h)
        cv2.line(frame, left_brow_inner, right_brow_inner, COLORS['neon_red'], 1)
        
        mouth_pts = [self.MOUTH_LEFT, self.MOUTH_RIGHT, self.LIP_TOP_CENTER, self.LIP_BOTTOM_CENTER]
        for idx in mouth_pts:
            pt = self._get_landmark_pos(landmarks, idx, w, h)
            cv2.circle(frame, pt, 4, COLORS['neon_purple'], -1)
        
        left_pos = self._get_landmark_pos(landmarks, self.MOUTH_LEFT, w, h)
        right_pos = self._get_landmark_pos(landmarks, self.MOUTH_RIGHT, w, h)
        center_y = int((landmarks[self.LIP_TOP_CENTER].y + landmarks[self.LIP_BOTTOM_CENTER].y) / 2 * h)
        
        cv2.line(frame, (left_pos[0], center_y), (right_pos[0], center_y), COLORS['white'], 1)
        cv2.line(frame, left_pos, right_pos, COLORS['neon_orange'], 1)
        
        return frame


class AirCanvas:
    def __init__(self, width, height):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_point = None
        self.drawing = False
        self.color_index = 0
        self.colors = [
            COLORS['neon_green'],
            COLORS['neon_pink'],
            COLORS['neon_blue'],
            COLORS['neon_yellow'],
            COLORS['neon_red'],
            COLORS['neon_cyan'],
            COLORS['white'],
        ]
        self.brush_size = 8
        self.points_buffer = deque(maxlen=5)
    
    @property
    def current_color(self):
        return self.colors[self.color_index]
    
    def next_color(self):
        self.color_index = (self.color_index + 1) % len(self.colors)
    
    def clear(self):
        self.canvas = np.zeros_like(self.canvas)
        self.prev_point = None
    
    def draw(self, point, is_drawing):
        if point is None:
            self.prev_point = None
            return
        
        self.points_buffer.append(point)
        
        if is_drawing:
            if len(self.points_buffer) >= 2:
                smoothed_point = (
                    int(np.mean([p[0] for p in self.points_buffer])),
                    int(np.mean([p[1] for p in self.points_buffer]))
                )
                
                if self.prev_point is not None:
                    cv2.line(self.canvas, self.prev_point, smoothed_point, 
                            self.current_color, self.brush_size, cv2.LINE_AA)
                
                self.prev_point = smoothed_point
        else:
            self.prev_point = None
    
    def get_overlay(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        glow = cv2.GaussianBlur(self.canvas, (21, 21), 0)
        result = frame.copy()
        result = cv2.addWeighted(result, 1, glow, 0.5, 0)
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch > 0, self.canvas, result)
        return result


class GestureRecognizer:
    def __init__(self):
        self.gesture_history = deque(maxlen=10)
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0
    
    def _fingers_up(self, hand_landmarks, handedness):
        fingers = []
        
        if handedness == "Right":
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)
        
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        
        for tip, pip in zip(tip_ids, pip_ids):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def recognize(self, hand_landmarks, handedness="Right"):
        if hand_landmarks is None:
            return None, []
        
        fingers = self._fingers_up(hand_landmarks, handedness)
        total_fingers = sum(fingers)
        
        gesture = None
        
        if fingers == [0, 1, 0, 0, 0]:
            gesture = "POINT"
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "PEACE"
        elif fingers == [1, 1, 1, 1, 1]:
            gesture = "OPEN"
        elif fingers == [0, 0, 0, 0, 0]:
            gesture = "FIST"
        elif fingers == [1, 0, 0, 0, 0]:
            gesture = "THUMBS_UP"
        elif fingers == [1, 1, 0, 0, 1]:
            gesture = "ROCK"
        elif fingers == [0, 1, 1, 1, 0]:
            gesture = "THREE"
        
        return gesture, fingers


class VisualEffects:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.matrix_drops = self._init_matrix_rain()
        self.glitch_intensity = 0
        self.clone_frames = deque(maxlen=10)
        self.effect_time = 0
    
    def _init_matrix_rain(self):
        drops = []
        for i in range(self.width // 20):
            drops.append({
                'x': i * 20,
                'y': random.randint(-self.height, 0),
                'speed': random.randint(5, 15),
                'chars': [chr(random.randint(0x30A0, 0x30FF)) for _ in range(random.randint(5, 15))]
            })
        return drops
    
    def matrix_rain(self, frame):
        overlay = frame.copy()
        overlay = cv2.addWeighted(overlay, 0.7, np.zeros_like(overlay), 0.3, 0)
        green_tint = np.zeros_like(overlay)
        green_tint[:, :, 1] = 30
        overlay = cv2.add(overlay, green_tint)
        
        for drop in self.matrix_drops:
            for i, char in enumerate(drop['chars']):
                y = drop['y'] + i * 20
                if 0 <= y < self.height:
                    brightness = max(0, 255 - i * 20)
                    color = (0, brightness, 0)
                    cv2.putText(overlay, char, (drop['x'], y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            drop['y'] += drop['speed']
            if drop['y'] > self.height:
                drop['y'] = random.randint(-200, 0)
                drop['chars'] = [chr(random.randint(0x30A0, 0x30FF)) for _ in range(random.randint(5, 15))]
        
        return overlay
    
    def glitch_effect(self, frame):
        result = frame.copy()
        
        if random.random() > 0.7:
            shift = random.randint(5, 20)
            direction = random.choice([-1, 1])
            b, g, r = cv2.split(result)
            
            if random.random() > 0.5:
                r = np.roll(r, shift * direction, axis=1)
            if random.random() > 0.5:
                b = np.roll(b, -shift * direction, axis=1)
            
            result = cv2.merge([b, g, r])
        
        if random.random() > 0.8:
            num_slices = random.randint(2, 5)
            for _ in range(num_slices):
                y = random.randint(0, self.height - 20)
                h = random.randint(5, 20)
                shift = random.randint(-30, 30)
                
                slice_img = result[y:y+h, :].copy()
                slice_img = np.roll(slice_img, shift, axis=1)
                result[y:y+h, :] = slice_img
        
        if random.random() > 0.9:
            x = random.randint(0, self.width - 100)
            y = random.randint(0, self.height - 30)
            w = random.randint(50, 150)
            h = random.randint(10, 30)
            color = random.choice([COLORS['neon_pink'], COLORS['neon_cyan'], COLORS['neon_green']])
            cv2.rectangle(result, (x, y), (x + w, y + h), color, -1)
        
        return result
    
    def neon_outline(self, frame, pose_landmarks):
        neon_canvas = np.zeros_like(frame)
        
        if pose_landmarks:
            h, w = frame.shape[:2]
            connections = mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start = pose_landmarks.landmark[start_idx]
                end = pose_landmarks.landmark[end_idx]
                pt1 = (int(start.x * w), int(start.y * h))
                pt2 = (int(end.x * w), int(end.y * h))
                cv2.line(neon_canvas, pt1, pt2, COLORS['neon_cyan'], 3)
            
            for landmark in pose_landmarks.landmark:
                pt = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(neon_canvas, pt, 5, COLORS['neon_pink'], -1)
        
        glow = cv2.GaussianBlur(neon_canvas, (25, 25), 0)
        neon_canvas = cv2.addWeighted(neon_canvas, 1, glow, 1, 0)
        dark_frame = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
        result = cv2.add(dark_frame, neon_canvas)
        
        return result
    
    def clone_effect(self, frame):
        self.clone_frames.append(frame.copy())
        
        if len(self.clone_frames) < 5:
            return frame
        
        result = frame.copy().astype(np.float32)
        
        for i, old_frame in enumerate(self.clone_frames):
            alpha = 0.15 * (i / len(self.clone_frames))
            result = cv2.addWeighted(result.astype(np.uint8), 1, old_frame, alpha, 0).astype(np.float32)
        
        return result.astype(np.uint8)
    
    def scanlines(self, frame):
        result = frame.copy()
        for y in range(0, self.height, 3):
            result[y, :] = result[y, :] * 0.7
        return result.astype(np.uint8)


class FocusMonitor:
    def __init__(self):
        self.focus_history = deque(maxlen=300)
        self.looking_away_start = None
        self.distraction_count = 0
        self.session_start = time.time()
        self.total_focused_time = 0
        self.last_focused = True
    
    def update(self, face_landmarks, frame_shape):
        if face_landmarks is None:
            if self.looking_away_start is None:
                self.looking_away_start = time.time()
            self.focus_history.append(0)
            focused = False
        else:
            h, w = frame_shape[:2]
            nose = face_landmarks.landmark[1]
            nose_x = nose.x
            center_threshold = 0.15
            if abs(nose_x - 0.5) < center_threshold:
                focused = True
                if self.looking_away_start is not None:
                    self.distraction_count += 1
                self.looking_away_start = None
            else:
                if self.looking_away_start is None:
                    self.looking_away_start = time.time()
                focused = False
            
            self.focus_history.append(1 if focused else 0)
        
        if focused and self.last_focused:
            self.total_focused_time += 1/30
        
        self.last_focused = focused
        return focused
    
    def get_focus_percentage(self):
        if len(self.focus_history) == 0:
            return 100.0
        return (sum(self.focus_history) / len(self.focus_history)) * 100
    
    def get_session_stats(self):
        session_duration = time.time() - self.session_start
        return {
            'duration': session_duration,
            'focused_time': self.total_focused_time,
            'focus_percentage': self.get_focus_percentage(),
            'distractions': self.distraction_count
        }
    
    def get_alert(self):
        if self.looking_away_start is not None:
            away_time = time.time() - self.looking_away_start
            if away_time > 3:
                return f"Voce desviou o olhar ha {away_time:.1f}s!"
        return None


class FitnessTracker:
    def __init__(self):
        self.squat_count = 0
        self.pushup_count = 0
        self.jumping_jack_count = 0
        self.squat_state = "up"
        self.pushup_state = "up"
        self.jj_state = "down"
        self.current_exercise = "squat"
        self.exercise_started = False
    
    def _calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def track_squat(self, pose_landmarks):
        if pose_landmarks is None:
            return
        lm = pose_landmarks.landmark
        hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        angle = self._calculate_angle(hip, knee, ankle)
        if angle < 100 and self.squat_state == "up":
            self.squat_state = "down"
        elif angle > 160 and self.squat_state == "down":
            self.squat_state = "up"
            self.squat_count += 1
        return angle
    
    def track_pushup(self, pose_landmarks):
        if pose_landmarks is None:
            return
        lm = pose_landmarks.landmark
        shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        angle = self._calculate_angle(shoulder, elbow, wrist)
        if angle < 90 and self.pushup_state == "up":
            self.pushup_state = "down"
        elif angle > 160 and self.pushup_state == "down":
            self.pushup_state = "up"
            self.pushup_count += 1
        return angle
    
    def get_count(self):
        if self.current_exercise == "squat":
            return self.squat_count
        elif self.current_exercise == "pushup":
            return self.pushup_count
        return 0
    
    def next_exercise(self):
        exercises = ["squat", "pushup"]
        idx = exercises.index(self.current_exercise)
        self.current_exercise = exercises[(idx + 1) % len(exercises)]
    
    def reset(self):
        self.squat_count = 0
        self.pushup_count = 0
        self.squat_state = "up"
        self.pushup_state = "up"


class VisionAIUltimate:
    def __init__(self):
        print("\n" + "=" * 70)
        print("Inicializando VISION AI ULTIMATE...")
        print("=" * 70)
        
        self.width = 1280
        self.height = 720
        
        print("Carregando YOLO v8...")
        self.yolo = YOLO('yolov8n.pt')
        
        print("Inicializando MediaPipe...")
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("Inicializando Emotion AI...")
        self.emotion_detector = EmotionDetector()
        
        print("Inicializando Air Canvas...")
        self.canvas = AirCanvas(self.width, self.height)
        
        print("Inicializando Gesture Recognition...")
        self.gesture_recognizer = GestureRecognizer()
        
        print("Inicializando Visual Effects...")
        self.effects = VisualEffects(self.width, self.height)
        
        print("Inicializando Focus Monitor...")
        self.focus_monitor = FocusMonitor()
        
        print("Inicializando Fitness Tracker...")
        self.fitness_tracker = FitnessTracker()
        
        self.current_mode = 0
        self.current_effect = 0
        self.effect_names = ['NONE', 'MATRIX', 'GLITCH', 'NEON', 'CLONE', 'SCANLINES']
        
        self.show_objects = True
        self.show_pose = True
        self.show_hands = True
        self.show_face = False
        
        self.fps = 0
        self.fps_counter = 0
        self.fps_start = time.time()
        
        self.emotion_graph_data = deque(maxlen=100)
        
        print("\nTodos os modulos carregados!")
        print("=" * 70 + "\n")
    
    def detect_objects(self, frame):
        results = self.yolo(frame, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo.names[cls]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['neon_green'], 2)
                
                text = f"{label} {conf:.0%}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 28), (x1 + w + 10, y1), COLORS['neon_green'], -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['black'], 2)
        
        return frame
    
    def process_hands(self, frame, rgb_frame):
        results = self.hands.process(rgb_frame)
        
        gesture = None
        index_finger_tip = None
        is_drawing = False
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if self.show_hands:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=COLORS['neon_yellow'], thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=COLORS['neon_green'], thickness=2)
                    )
                
                handedness = "Right"
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                
                gesture, fingers = self.gesture_recognizer.recognize(hand_landmarks, handedness)
                
                index_tip = hand_landmarks.landmark[8]
                index_finger_tip = (
                    int(index_tip.x * self.width),
                    int(index_tip.y * self.height)
                )
                
                is_drawing = gesture == "POINT"
        
        return frame, gesture, index_finger_tip, is_drawing
    
    def process_pose(self, frame, rgb_frame):
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks and self.show_pose:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=COLORS['neon_pink'], thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=COLORS['neon_blue'], thickness=2)
            )
        
        return frame, results.pose_landmarks if results.pose_landmarks else None
    
    def process_face(self, frame, rgb_frame):
        results = self.face_mesh.process(rgb_frame)
        
        face_landmarks = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            if self.show_face:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        
        return frame, face_landmarks
    
    def draw_emotion_panel(self, frame, emotion, confidence):
        h, w = frame.shape[:2]
        
        panel_w = 300
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 80), (w, 550), COLORS['dark_gray'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.putText(frame, "EMOTION AI PRO", (w - panel_w + 20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['neon_cyan'], 2)
        
        emotion_info = EMOTIONS.get(emotion, EMOTIONS['neutral'])
        cv2.putText(frame, emotion_info['label'], (w - panel_w + 20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, emotion_info['color'], 2)
        
        bar_x = w - panel_w + 20
        bar_y = 175
        bar_w = panel_w - 40
        bar_h = 15
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLORS['gray'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + bar_h), 
                     emotion_info['color'], -1)
        cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w - 35, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['white'], 1)
        
        debug = self.emotion_detector.debug_info
        if debug:
            y_offset = 210
            
            cv2.putText(frame, "BOCA", (w - panel_w + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['neon_pink'], 1)
            
            mouth_curve = debug.get('mouth_curve', 0)
            curve_color = COLORS['neon_green'] if mouth_curve > 2 else COLORS['neon_red'] if mouth_curve < -2 else COLORS['white']
            self._draw_metric_bar(frame, w - panel_w + 20, y_offset + 15, "Curvatura", mouth_curve, -10, 10, curve_color)
            
            mouth_open = debug.get('mouth_open', 0)
            self._draw_metric_bar(frame, w - panel_w + 20, y_offset + 35, "Abertura", mouth_open, 0, 50, COLORS['neon_cyan'])
            
            y_offset = 280
            cv2.putText(frame, "OLHOS", (w - panel_w + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['neon_green'], 1)
            
            eye_open = debug.get('eye_open', 0)
            eye_color = COLORS['neon_yellow'] if eye_open > 1 else COLORS['white']
            self._draw_metric_bar(frame, w - panel_w + 20, y_offset + 15, "Abertura", eye_open, -3, 3, eye_color)
            
            y_offset = 330
            cv2.putText(frame, "SOBRANCELHAS", (w - panel_w + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['neon_yellow'], 1)
            
            brow_h = debug.get('brow_height', 0)
            brow_color = COLORS['neon_yellow'] if brow_h > 3 else COLORS['neon_red'] if brow_h < -3 else COLORS['white']
            self._draw_metric_bar(frame, w - panel_w + 20, y_offset + 15, "Altura", brow_h, -10, 10, brow_color)
            
            brow_d = debug.get('brow_distance', 0)
            dist_color = COLORS['neon_red'] if brow_d < -5 else COLORS['white']
            self._draw_metric_bar(frame, w - panel_w + 20, y_offset + 35, "Distancia", brow_d, -15, 15, dist_color)
            
            y_offset = 400
            cv2.putText(frame, "SCORES", (w - panel_w + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
            
            scores = debug.get('scores', {})
            y_score = y_offset + 18
            for emo, score in sorted(scores.items(), key=lambda x: -x[1])[:4]:
                emo_info = EMOTIONS.get(emo, EMOTIONS['neutral'])
                score_w = int(score * 1.5)
                cv2.rectangle(frame, (bar_x, y_score), (bar_x + score_w, y_score + 12), emo_info['color'], -1)
                cv2.putText(frame, f"{emo[:3].upper()}", (bar_x + score_w + 5, y_score + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, emo_info['color'], 1)
                y_score += 18
        
        self.emotion_graph_data.append(emotion)
        
        graph_y = 490
        graph_h = 40
        
        cv2.putText(frame, "Historico", (w - panel_w + 20, graph_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['gray'], 1)
        
        if len(self.emotion_graph_data) > 1:
            bar_width = max(1, (panel_w - 40) // len(self.emotion_graph_data))
            for i, em in enumerate(self.emotion_graph_data):
                em_info = EMOTIONS.get(em, EMOTIONS['neutral'])
                x = bar_x + i * bar_width
                cv2.rectangle(frame, (x, graph_y + 12), (x + bar_width - 1, graph_y + graph_h),
                             em_info['color'], -1)
        
        return frame
    
    def _draw_metric_bar(self, frame, x, y, label, value, min_val, max_val, color):
        h, w = frame.shape[:2]
        bar_w = 150
        
        cv2.putText(frame, f"{label}:", (x, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['gray'], 1)
        
        bar_x = x + 70
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 12), COLORS['dark_gray'], -1)
        
        center = bar_x + bar_w // 2
        cv2.line(frame, (center, y), (center, y + 12), COLORS['gray'], 1)
        
        normalized = (value - min_val) / (max_val - min_val + 0.001)
        normalized = max(0, min(1, normalized))
        marker_x = int(bar_x + normalized * bar_w)
        
        cv2.circle(frame, (marker_x, y + 6), 5, color, -1)
        
        cv2.putText(frame, f"{value:.1f}", (bar_x + bar_w + 5, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    def draw_canvas_ui(self, frame):
        h, w = frame.shape[:2]
        
        panel_x = 20
        panel_y = 100
        
        cv2.putText(frame, "AIR CANVAS", (panel_x, panel_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['neon_cyan'], 2)
        
        for i, color in enumerate(self.canvas.colors):
            y = panel_y + i * 40
            cv2.rectangle(frame, (panel_x, y), (panel_x + 30, y + 30), color, -1)
            
            if i == self.canvas.color_index:
                cv2.rectangle(frame, (panel_x - 3, y - 3), (panel_x + 33, y + 33), 
                             COLORS['white'], 2)
        
        cv2.putText(frame, "Aponte para desenhar", (panel_x, panel_y + len(self.canvas.colors) * 40 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        cv2.putText(frame, "Punho = limpar", (panel_x, panel_y + len(self.canvas.colors) * 40 + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        cv2.putText(frame, "Peace = trocar cor", (panel_x, panel_y + len(self.canvas.colors) * 40 + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        
        return frame
    
    def draw_focus_panel(self, frame, is_focused):
        h, w = frame.shape[:2]
        
        stats = self.focus_monitor.get_session_stats()
        
        panel_w = 280
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 80), (w, 320), COLORS['dark_gray'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "FOCUS MONITOR", (w - panel_w + 20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['neon_cyan'], 2)
        
        status_color = COLORS['neon_green'] if is_focused else COLORS['neon_red']
        status_text = "FOCADO" if is_focused else "DISTRAIDO"
        cv2.circle(frame, (w - panel_w + 30, 150), 10, status_color, -1)
        cv2.putText(frame, status_text, (w - panel_w + 50, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        y_offset = 190
        cv2.putText(frame, f"Tempo focado: {stats['focused_time']:.0f}s", 
                   (w - panel_w + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        cv2.putText(frame, f"Foco: {stats['focus_percentage']:.1f}%", 
                   (w - panel_w + 20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        cv2.putText(frame, f"Distracoes: {stats['distractions']}", 
                   (w - panel_w + 20, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        
        bar_x = w - panel_w + 20
        bar_y = y_offset + 70
        bar_w = panel_w - 40
        bar_h = 25
        
        focus_pct = stats['focus_percentage'] / 100
        bar_color = COLORS['neon_green'] if focus_pct > 0.7 else COLORS['neon_yellow'] if focus_pct > 0.4 else COLORS['neon_red']
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLORS['gray'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * focus_pct), bar_y + bar_h), bar_color, -1)
        
        alert = self.focus_monitor.get_alert()
        if alert:
            cv2.putText(frame, alert, (w - panel_w + 20, bar_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['neon_red'], 1)
        
        return frame
    
    def draw_fitness_panel(self, frame, pose_landmarks):
        h, w = frame.shape[:2]
        
        if pose_landmarks:
            if self.fitness_tracker.current_exercise == "squat":
                angle = self.fitness_tracker.track_squat(pose_landmarks)
            else:
                angle = self.fitness_tracker.track_pushup(pose_landmarks)
        
        panel_w = 250
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 80), (w, 300), COLORS['dark_gray'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "FITNESS TRACKER", (w - panel_w + 20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['neon_cyan'], 2)
        
        exercise_name = "AGACHAMENTO" if self.fitness_tracker.current_exercise == "squat" else "FLEXAO"
        cv2.putText(frame, exercise_name, (w - panel_w + 20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['neon_yellow'], 2)
        
        count = self.fitness_tracker.get_count()
        cv2.putText(frame, str(count), (w - panel_w + 80, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, COLORS['neon_green'], 4)
        cv2.putText(frame, "reps", (w - panel_w + 160, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 1)
        
        cv2.putText(frame, "[E] Trocar exercicio", (w - panel_w + 20, 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['gray'], 1)
        cv2.putText(frame, "[R] Resetar contagem", (w - panel_w + 20, 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['gray'], 1)
        
        return frame
    
    def draw_hud(self, frame, gesture=None):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), COLORS['black'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "VISION AI ULTIMATE", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['neon_cyan'], 2)
        
        mode_name = MODES[self.current_mode]
        cv2.putText(frame, f"Mode: {mode_name}", (20, 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['neon_green'], 1)
        
        fps_color = COLORS['neon_green'] if self.fps > 20 else COLORS['neon_yellow'] if self.fps > 10 else COLORS['neon_red']
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w - 100, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        if self.current_effect > 0:
            cv2.putText(frame, f"Effect: {self.effect_names[self.current_effect]}", 
                       (w - 200, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['neon_pink'], 1)
        
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (w // 2 - 80, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['neon_yellow'], 2)
        
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 50), (w, h), COLORS['black'], -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        
        controls = "[1-6] Modos | [E] Effect | [C] Clear | [Q] Quit"
        cv2.putText(frame, controls, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['gray'], 1)
        
        modules = [
            ("OBJ", self.show_objects),
            ("POSE", self.show_pose),
            ("HAND", self.show_hands),
            ("FACE", self.show_face),
        ]
        
        x_offset = w - 350
        for name, active in modules:
            color = COLORS['neon_green'] if active else COLORS['gray']
            cv2.putText(frame, name, (x_offset, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            x_offset += 60
        
        return frame
    
    def apply_effects(self, frame, pose_landmarks=None):
        if self.current_effect == 0:
            return frame
        elif self.current_effect == 1:
            return self.effects.matrix_rain(frame)
        elif self.current_effect == 2:
            return self.effects.glitch_effect(frame)
        elif self.current_effect == 3:
            return self.effects.neon_outline(frame, pose_landmarks)
        elif self.current_effect == 4:
            return self.effects.clone_effect(frame)
        elif self.current_effect == 5:
            return self.effects.scanlines(frame)
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not cap.isOpened():
            print("Erro: Nao foi possivel abrir a camera!")
            return
        
        print("\n" + "=" * 70)
        print("VISION AI ULTIMATE - INICIADO!")
        print("=" * 70)
        print("\nCONTROLES:")
        print("   [1] Modo Detection  [2] Modo Emotion   [3] Modo Canvas")
        print("   [4] Modo Effects    [5] Modo Focus     [6] Modo Fitness")
        print("   [E] Trocar Efeito   [C] Limpar Canvas  [R] Reset Fitness")
        print("   [O] Toggle Objects  [P] Toggle Pose    [H] Toggle Hands")
        print("   [F] Toggle Face     [Q] Sair")
        print("\n" + "=" * 70 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame, gesture, finger_pos, is_drawing = self.process_hands(frame, rgb_frame)
            frame, pose_landmarks = self.process_pose(frame, rgb_frame)
            frame, face_landmarks = self.process_face(frame, rgb_frame)
            
            if self.show_objects and self.current_mode == 0:
                frame = self.detect_objects(frame)
            
            if self.current_mode == 1:
                emotion, confidence, emotion_landmarks = self.emotion_detector.detect(frame)
                frame = self.emotion_detector.draw_face_debug(frame, emotion_landmarks)
                frame = self.draw_emotion_panel(frame, emotion, confidence)
            
            elif self.current_mode == 2:
                if gesture == "FIST":
                    self.canvas.clear()
                elif gesture == "PEACE":
                    self.canvas.next_color()
                
                self.canvas.draw(finger_pos, is_drawing)
                frame = self.canvas.get_overlay(frame)
                frame = self.draw_canvas_ui(frame)
            
            elif self.current_mode == 3:
                frame = self.apply_effects(frame, pose_landmarks)
            
            elif self.current_mode == 4:
                is_focused = self.focus_monitor.update(face_landmarks, frame.shape)
                frame = self.draw_focus_panel(frame, is_focused)
            
            elif self.current_mode == 5:
                frame = self.draw_fitness_panel(frame, pose_landmarks)
            
            if self.current_mode != 3 and self.current_effect > 0:
                frame = self.apply_effects(frame, pose_landmarks)
            
            frame = self.draw_hud(frame, gesture)
            
            self.fps_counter += 1
            if time.time() - self.fps_start >= 1:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start = time.time()
            
            cv2.imshow('Vision AI Ultimate', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.current_mode = 0
                print("Modo: DETECTION")
            elif key == ord('2'):
                self.current_mode = 1
                print("Modo: EMOTION")
            elif key == ord('3'):
                self.current_mode = 2
                print("Modo: CANVAS")
            elif key == ord('4'):
                self.current_mode = 3
                print("Modo: EFFECTS")
            elif key == ord('5'):
                self.current_mode = 4
                print("Modo: FOCUS")
            elif key == ord('6'):
                self.current_mode = 5
                print("Modo: FITNESS")
            elif key == ord('e'):
                self.current_effect = (self.current_effect + 1) % len(self.effect_names)
                print(f"Efeito: {self.effect_names[self.current_effect]}")
            elif key == ord('c'):
                self.canvas.clear()
                print("Canvas limpo!")
            elif key == ord('r'):
                self.fitness_tracker.reset()
                print("Fitness resetado!")
            elif key == ord('o'):
                self.show_objects = not self.show_objects
                print(f"Objects: {'ON' if self.show_objects else 'OFF'}")
            elif key == ord('p'):
                self.show_pose = not self.show_pose
                print(f"Pose: {'ON' if self.show_pose else 'OFF'}")
            elif key == ord('h'):
                self.show_hands = not self.show_hands
                print(f"Hands: {'ON' if self.show_hands else 'OFF'}")
            elif key == ord('f'):
                self.show_face = not self.show_face
                print(f"Face: {'ON' if self.show_face else 'OFF'}")
            elif key == ord('x'):
                self.fitness_tracker.next_exercise()
                print(f"Exercicio: {self.fitness_tracker.current_exercise}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.current_mode == 4:
            stats = self.focus_monitor.get_session_stats()
            print("\n" + "=" * 70)
            print("ESTATISTICAS DA SESSAO DE FOCO")
            print("=" * 70)
            print(f"   Duracao: {stats['duration']:.0f}s")
            print(f"   Tempo focado: {stats['focused_time']:.0f}s")
            print(f"   Porcentagem de foco: {stats['focus_percentage']:.1f}%")
            print(f"   Numero de distracoes: {stats['distractions']}")
            print("=" * 70)
        
        print("\nVision AI Ultimate encerrado!\n")


if __name__ == "__main__":
    print("""
    
    VISION AI ULTIMATE
    
    """)
    
    vision = VisionAIUltimate()
    vision.run()
