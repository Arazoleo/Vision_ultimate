import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

COLORS = {
    'bg_dark': (15, 15, 25),
    'primary': (255, 107, 107),
    'secondary': (78, 205, 196),
    'accent': (255, 230, 109),
    'success': (107, 255, 148),
    'text': (255, 255, 255),
    'text_dim': (150, 150, 170),
    'panel': (30, 30, 45),
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class LibrasInterpreter:
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        self.current_letter = None
        self.letter_history = deque(maxlen=15)
        self.confirmed_letter = None
        self.last_confirmed_time = 0
        self.confirmation_threshold = 0.6
        
        self.word_buffer = []
        self.sentence = ""
        
        self.stable_count = 0
        self.last_stable_letter = None
        
    def _get_finger_states(self, hand_landmarks, handedness="Right"):
        landmarks = hand_landmarks.landmark
        fingers = []
        
        if handedness == "Right":
            thumb_open = landmarks[4].x < landmarks[3].x
        else:
            thumb_open = landmarks[4].x > landmarks[3].x
        fingers.append(1 if thumb_open else 0)
        
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        for tip, pip in zip(tips, pips):
            fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
        
        return fingers
    
    def _get_finger_angles(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        def angle_3points(a, b, c):
            ba = np.array([a.x - b.x, a.y - b.y])
            bc = np.array([c.x - b.x, c.y - b.y])
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 0.001)
            return np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        
        angles = {
            'index': angle_3points(landmarks[5], landmarks[6], landmarks[8]),
            'middle': angle_3points(landmarks[9], landmarks[10], landmarks[12]),
            'ring': angle_3points(landmarks[13], landmarks[14], landmarks[16]),
            'pinky': angle_3points(landmarks[17], landmarks[18], landmarks[20]),
        }
        
        return angles
    
    def _get_hand_orientation(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
        
        palm_vector = np.cross(index_mcp - wrist, pinky_mcp - wrist)
        palm_facing = palm_vector[2] > 0
        hand_tilt = landmarks[12].x - landmarks[0].x
        
        return {
            'palm_facing': palm_facing,
            'tilt': hand_tilt,
            'wrist_y': landmarks[0].y,
            'fingers_up': landmarks[12].y < landmarks[0].y
        }
    
    def _detect_letter(self, hand_landmarks, handedness="Right"):
        fingers = self._get_finger_states(hand_landmarks, handedness)
        orientation = self._get_hand_orientation(hand_landmarks)
        landmarks = hand_landmarks.landmark
        
        thumb, index, middle, ring, pinky = fingers
        
        thumb_index_dist = np.sqrt(
            (landmarks[4].x - landmarks[8].x)**2 + 
            (landmarks[4].y - landmarks[8].y)**2
        )
        
        thumb_middle_dist = np.sqrt(
            (landmarks[4].x - landmarks[12].x)**2 + 
            (landmarks[4].y - landmarks[12].y)**2
        )
        
        index_middle_dist = np.sqrt(
            (landmarks[8].x - landmarks[12].x)**2 + 
            (landmarks[8].y - landmarks[12].y)**2
        )
        
        if fingers == [1, 0, 0, 0, 0]:
            if landmarks[4].y > landmarks[3].y:
                return 'A'
        
        if fingers == [0, 1, 1, 1, 1]:
            return 'B'
        
        if thumb and index and middle and ring and pinky:
            if 0.1 < thumb_index_dist < 0.25:
                return 'C'
        
        if fingers == [1, 1, 0, 0, 0]:
            if thumb_middle_dist < 0.08:
                return 'D'
        
        if fingers == [0, 0, 0, 0, 0]:
            if landmarks[8].y < landmarks[5].y + 0.05:
                return 'E'
        
        if fingers == [1, 0, 1, 1, 1]:
            if thumb_index_dist < 0.06:
                return 'F'
        
        if fingers == [1, 1, 0, 0, 0]:
            if abs(landmarks[8].x - landmarks[5].x) > 0.1:
                return 'G'
        
        if fingers == [0, 1, 1, 0, 0]:
            if abs(landmarks[8].x - landmarks[5].x) > 0.08:
                return 'H'
        
        if fingers == [0, 0, 0, 0, 1]:
            return 'I'
        
        if fingers == [0, 0, 0, 0, 1]:
            if orientation['tilt'] > 0.1:
                return 'J'
        
        if fingers == [1, 1, 1, 0, 0]:
            if landmarks[4].y > landmarks[6].y and landmarks[4].y < landmarks[8].y:
                return 'K'
        
        if fingers == [1, 1, 0, 0, 0]:
            if thumb_index_dist > 0.15:
                return 'L'
        
        if fingers == [0, 0, 0, 0, 0]:
            if landmarks[4].y > landmarks[8].y:
                return 'M'
        
        if fingers == [0, 0, 0, 0, 0]:
            if landmarks[4].y > landmarks[8].y and landmarks[4].y < landmarks[12].y:
                return 'N'
        
        if 0.02 < thumb_index_dist < 0.08:
            if not middle and not ring and not pinky:
                return 'O'
        
        if fingers == [1, 1, 1, 0, 0]:
            if landmarks[8].y > landmarks[5].y:
                return 'P'
        
        if fingers == [1, 1, 0, 0, 0]:
            if landmarks[8].y > landmarks[0].y:
                return 'Q'
        
        if index and middle and not ring and not pinky:
            if abs(landmarks[8].x - landmarks[12].x) < 0.03:
                return 'R'
        
        if fingers == [0, 0, 0, 0, 0]:
            if landmarks[4].y < landmarks[8].y:
                return 'S'
        
        if fingers == [0, 0, 0, 0, 0]:
            if landmarks[4].x > landmarks[5].x and landmarks[4].x < landmarks[9].x:
                return 'T'
        
        if fingers == [0, 1, 1, 0, 0]:
            if index_middle_dist < 0.04:
                return 'U'
        
        if fingers == [0, 1, 1, 0, 0]:
            if index_middle_dist > 0.06:
                return 'V'
        
        if fingers == [0, 1, 1, 1, 0]:
            return 'W'
        
        if fingers == [0, 0, 0, 0, 0]:
            if landmarks[8].y < landmarks[7].y and landmarks[8].y > landmarks[5].y:
                return 'X'
        
        if fingers == [1, 0, 0, 0, 1]:
            return 'Y'
        
        if fingers == [0, 1, 0, 0, 0]:
            return 'Z'
        
        return None
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_letter = None
        hand_landmarks = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = "Right"
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            
            detected_letter = self._detect_letter(hand_landmarks, handedness)
            
            if detected_letter:
                self.letter_history.append(detected_letter)
                
                if len(self.letter_history) >= 8:
                    from collections import Counter
                    most_common = Counter(self.letter_history).most_common(1)[0]
                    if most_common[1] >= 6:
                        self.current_letter = most_common[0]
                        
                        current_time = time.time()
                        if (self.current_letter != self.confirmed_letter or 
                            current_time - self.last_confirmed_time > 1.5):
                            
                            if self.stable_count >= 10:
                                self.confirmed_letter = self.current_letter
                                self.last_confirmed_time = current_time
                                self.word_buffer.append(self.confirmed_letter)
                                self.stable_count = 0
                            else:
                                if self.current_letter == self.last_stable_letter:
                                    self.stable_count += 1
                                else:
                                    self.stable_count = 0
                                    self.last_stable_letter = self.current_letter
        else:
            self.current_letter = None
        
        return self.current_letter, hand_landmarks
    
    def get_word(self):
        return ''.join(self.word_buffer)
    
    def confirm_word(self):
        if self.word_buffer:
            word = ''.join(self.word_buffer)
            self.sentence += word + " "
            self.word_buffer = []
            return word
        return None
    
    def clear_word(self):
        self.word_buffer = []
    
    def backspace(self):
        if self.word_buffer:
            self.word_buffer.pop()


class LibrasApp:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.interpreter = LibrasInterpreter()
        
        self.show_landmarks = True
        self.show_debug = False
        
    def draw_hand_landmarks(self, frame, landmarks):
        if landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        for start, end in connections:
            pt1 = (int(landmarks.landmark[start].x * w), 
                   int(landmarks.landmark[start].y * h))
            pt2 = (int(landmarks.landmark[end].x * w), 
                   int(landmarks.landmark[end].y * h))
            
            cv2.line(frame, pt1, pt2, COLORS['secondary'], 6)
            cv2.line(frame, pt1, pt2, COLORS['text'], 2)
        
        for i, lm in enumerate(landmarks.landmark):
            pt = (int(lm.x * w), int(lm.y * h))
            
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(frame, pt, 10, COLORS['primary'], -1)
                cv2.circle(frame, pt, 12, COLORS['text'], 2)
            else:
                cv2.circle(frame, pt, 5, COLORS['secondary'], -1)
        
        return frame
    
    def draw_ui(self, frame, current_letter):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        overlay = frame.copy()
        
        cv2.putText(frame, "LIBRAS INTERPRETER", (30, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['primary'], 2)
        cv2.putText(frame, "Alfabeto Manual Brasileiro", (30, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_dim'], 1)
        
        letter_panel_x = w - 250
        cv2.rectangle(overlay, (letter_panel_x, 100), (w - 20, 300), COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        overlay = frame.copy()
        
        cv2.putText(frame, "LETRA DETECTADA", (letter_panel_x + 20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dim'], 1)
        
        if current_letter:
            cv2.putText(frame, current_letter, (letter_panel_x + 70, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 4, COLORS['accent'], 4)
            
            confidence = min(self.interpreter.stable_count / 10, 1.0)
            bar_w = 180
            bar_h = 8
            bar_x = letter_panel_x + 25
            bar_y = 270
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                         COLORS['text_dim'], -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + bar_h), 
                         COLORS['success'] if confidence >= 1 else COLORS['secondary'], -1)
        else:
            cv2.putText(frame, "?", (letter_panel_x + 80, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 4, COLORS['text_dim'], 3)
        
        word = self.interpreter.get_word()
        cv2.rectangle(overlay, (letter_panel_x, 320), (w - 20, 420), COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        overlay = frame.copy()
        
        cv2.putText(frame, "PALAVRA", (letter_panel_x + 20, 355),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dim'], 1)
        
        display_word = word[-12:] if len(word) > 12 else word
        if display_word:
            cv2.putText(frame, display_word, (letter_panel_x + 20, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['text'], 2)
        else:
            cv2.putText(frame, "...", (letter_panel_x + 20, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['text_dim'], 2)
        
        sentence = self.interpreter.sentence
        if sentence:
            cv2.rectangle(overlay, (20, h - 100), (w - 270, h - 20), COLORS['panel'], -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            overlay = frame.copy()
            
            cv2.putText(frame, "Frase:", (35, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text_dim'], 1)
            
            display_sentence = sentence[-50:] if len(sentence) > 50 else sentence
            cv2.putText(frame, display_sentence, (35, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 1)
        
        controls_y = h - 30
        controls = "[SPACE] Confirmar palavra | [BACKSPACE] Apagar | [C] Limpar tudo | [Q] Sair"
        cv2.putText(frame, controls, (20, controls_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['text_dim'], 1)
        
        ref_y = 450
        cv2.putText(frame, "ALFABETO:", (letter_panel_x + 20, ref_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_dim'], 1)
        
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, letter in enumerate(alphabet):
            row = i // 9
            col = i % 9
            x = letter_panel_x + 20 + col * 24
            y = ref_y + 25 + row * 25
            
            color = COLORS['accent'] if letter == current_letter else COLORS['text_dim']
            cv2.putText(frame, letter, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not cap.isOpened():
            print("Erro: Nao foi possivel abrir a camera!")
            return
        
        print("\n" + "=" * 60)
        print("  LIBRAS INTERPRETER - Alfabeto Manual")
        print("=" * 60)
        print("\nControles:")
        print("  [SPACE]     - Confirmar palavra")
        print("  [BACKSPACE] - Apagar ultima letra")
        print("  [C]         - Limpar tudo")
        print("  [L]         - Toggle landmarks")
        print("  [Q]         - Sair")
        print("\nMostre as letras do alfabeto manual de Libras!")
        print("=" * 60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            current_letter, hand_landmarks = self.interpreter.process_frame(frame)
            
            if self.show_landmarks and hand_landmarks:
                frame = self.draw_hand_landmarks(frame, hand_landmarks)
            
            frame = self.draw_ui(frame, current_letter)
            
            cv2.imshow('Libras Interpreter', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                word = self.interpreter.confirm_word()
                if word:
                    print(f"Palavra confirmada: {word}")
            elif key == 8:
                self.interpreter.backspace()
            elif key == ord('c'):
                self.interpreter.clear_word()
                self.interpreter.sentence = ""
                print("Limpo!")
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.interpreter.sentence or self.interpreter.word_buffer:
            final = self.interpreter.sentence + self.interpreter.get_word()
            print(f"\n{'=' * 60}")
            print(f"Texto final: {final}")
            print(f"{'=' * 60}\n")


if __name__ == "__main__":
    app = LibrasApp()
    app.run()
