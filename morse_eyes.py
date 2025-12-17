import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

COLORS = {
    'bg': (20, 18, 25),
    'primary': (255, 180, 100),
    'secondary': (100, 180, 255),
    'accent': (150, 255, 200),
    'dot': (100, 255, 180),
    'dash': (180, 150, 255),
    'eye_open': (100, 255, 150),
    'eye_closed': (100, 150, 255),
    'warning': (80, 200, 255),
    'text': (255, 255, 255),
    'dim': (100, 100, 130),
    'panel': (30, 28, 40),
    'success': (100, 255, 180),
}

MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9', '-----': '0',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'",
    '-.-.--': '!', '-..-.': '/', '-.--.': '(', '-.--.-': ')',
}

CHAR_TO_MORSE = {v: k for k, v in MORSE_CODE.items()}

mp_face_mesh = mp.solutions.face_mesh


class EyeTracker:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_EYE_TOP = 386
        self.LEFT_EYE_BOTTOM = 374
        self.LEFT_EYE_LEFT = 263
        self.LEFT_EYE_RIGHT = 362
        
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_TOP = 159
        self.RIGHT_EYE_BOTTOM = 145
        self.RIGHT_EYE_LEFT = 33
        self.RIGHT_EYE_RIGHT = 133
        
        self.ear_history = deque(maxlen=10)
        self.blink_threshold = 0.21
        self.eyes_closed = False
        self.eyes_closed_start = None
        
        self.calibration_data = deque(maxlen=60)
        self.calibrated = False
        self.baseline_ear = 0.25
        
    def calculate_ear(self, landmarks, eye_points, w, h):
        def get_point(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        
        if len(eye_points) == 6:
            p1 = get_point(eye_points[0])
            p2 = get_point(eye_points[1])
            p3 = get_point(eye_points[2])
            p4 = get_point(eye_points[3])
            p5 = get_point(eye_points[4])
            p6 = get_point(eye_points[5])
            
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            h_dist = np.linalg.norm(p1 - p4)
            
            ear = (v1 + v2) / (2.0 * h_dist + 0.001)
            return ear
        
        return 0.25
    
    def process(self, frame):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None, False, 0
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE, w, h)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2
        
        if not self.calibrated:
            self.calibration_data.append(avg_ear)
            if len(self.calibration_data) >= 30:
                sorted_ears = sorted(self.calibration_data)
                self.baseline_ear = sorted_ears[int(len(sorted_ears) * 0.75)]
                self.blink_threshold = self.baseline_ear * 0.7
                self.calibrated = True
        
        self.ear_history.append(avg_ear)
        eyes_currently_closed = avg_ear < self.blink_threshold
        
        eye_positions = {
            'left': self._get_eye_bbox(landmarks, self.LEFT_EYE, w, h),
            'right': self._get_eye_bbox(landmarks, self.RIGHT_EYE, w, h),
        }
        
        return landmarks, eye_positions, eyes_currently_closed, avg_ear
    
    def _get_eye_bbox(self, landmarks, eye_points, w, h):
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                  for i in eye_points]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return {
            'x': min(xs) - 10,
            'y': min(ys) - 10,
            'w': max(xs) - min(xs) + 20,
            'h': max(ys) - min(ys) + 20,
            'center': ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
        }


class MorseEyesInterpreter:
    def __init__(self):
        self.eye_tracker = EyeTracker()
        
        self.eyes_closed = False
        self.close_start_time = None
        self.last_open_time = time.time()
        
        self.DOT_MAX = 0.25
        self.DASH_MIN = 0.4
        self.LETTER_TIMEOUT = 1.5
        self.WORD_TIMEOUT = 2.5
        
        self.morse_buffer = []
        self.word_buffer = []
        self.sentence = ""
        
        self.last_input_time = time.time()
        self.last_action = None
        self.action_time = 0
        
        self.blink_animation = 0
        self.particles = []
        self.blink_times = deque(maxlen=5)
        
    def process_frame(self, frame):
        landmarks, eye_positions, eyes_closed, ear = self.eye_tracker.process(frame)
        
        current_time = time.time()
        action = None
        
        if landmarks is None:
            return None, None, None, 0, "SEM ROSTO"
        
        if not self.eye_tracker.calibrated:
            return landmarks, eye_positions, None, ear, "CALIBRANDO..."
        
        if eyes_closed and not self.eyes_closed:
            self.close_start_time = current_time
            self.eyes_closed = True
        
        elif not eyes_closed and self.eyes_closed:
            if self.close_start_time:
                blink_duration = current_time - self.close_start_time
                self.blink_times.append(current_time)
                
                if blink_duration >= 0.08:
                    if blink_duration < self.DOT_MAX:
                        self.morse_buffer.append('.')
                        action = "PONTO (.)"
                        self._add_particles("dot")
                        self.blink_animation = 1.0
                    
                    elif blink_duration >= self.DASH_MIN:
                        self.morse_buffer.append('-')
                        action = "TRACO (-)"
                        self._add_particles("dash")
                        self.blink_animation = 1.0
                    
                    self.last_input_time = current_time
            
            self.eyes_closed = False
            self.close_start_time = None
            self.last_open_time = current_time
        
        if len(self.blink_times) >= 2:
            if self.blink_times[-1] - self.blink_times[-2] < 0.4:
                if self.morse_buffer:
                    morse = ''.join(self.morse_buffer)
                    char = MORSE_CODE.get(morse, '?')
                    self.word_buffer.append(char)
                    self.morse_buffer = []
                    action = f"LETRA: {char}"
                    self.blink_times.clear()
                    self.last_input_time = current_time
        
        if eyes_closed and self.close_start_time:
            closed_duration = current_time - self.close_start_time
            if closed_duration > self.WORD_TIMEOUT:
                if self.morse_buffer:
                    morse = ''.join(self.morse_buffer)
                    char = MORSE_CODE.get(morse, '?')
                    self.word_buffer.append(char)
                    self.morse_buffer = []
                
                if self.word_buffer:
                    word = ''.join(self.word_buffer)
                    self.sentence += word + " "
                    self.word_buffer = []
                    action = f"PALAVRA: {word}"
                
                self.close_start_time = None
                self.last_input_time = current_time
        
        if (self.morse_buffer and 
            not eyes_closed and
            current_time - self.last_input_time > self.LETTER_TIMEOUT):
            morse = ''.join(self.morse_buffer)
            char = MORSE_CODE.get(morse, '?')
            self.word_buffer.append(char)
            self.morse_buffer = []
            action = f"AUTO: {char}"
            self.last_input_time = current_time
        
        if self.blink_animation > 0:
            self.blink_animation -= 0.05
        
        self._update_particles()
        
        if eyes_closed:
            if self.close_start_time:
                duration = current_time - self.close_start_time
                if duration > self.WORD_TIMEOUT:
                    status = "CONFIRMANDO PALAVRA..."
                elif duration > self.DASH_MIN:
                    status = "TRACO (-)"
                elif duration > self.DOT_MAX:
                    status = "..."
                else:
                    status = "PISCANDO"
            else:
                status = "OLHOS FECHADOS"
        else:
            status = "PRONTO"
        
        return landmarks, eye_positions, action, ear, status
    
    def _add_particles(self, ptype):
        for _ in range(8):
            self.particles.append({
                'x': 640 + np.random.randint(-100, 100),
                'y': 200 + np.random.randint(-50, 50),
                'vx': np.random.randn() * 4,
                'vy': np.random.randn() * 4 - 2,
                'life': 1.0,
                'type': ptype
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 0.03
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def get_morse_code(self):
        return ''.join(self.morse_buffer)
    
    def get_word(self):
        return ''.join(self.word_buffer)
    
    def clear_all(self):
        self.morse_buffer = []
        self.word_buffer = []
        self.sentence = ""
    
    def backspace(self):
        if self.morse_buffer:
            self.morse_buffer.pop()
        elif self.word_buffer:
            self.word_buffer.pop()


class MorseEyesApp:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.interpreter = MorseEyesInterpreter()
        
        self.show_face_mesh = False
        self.show_reference = True
        self.last_action = None
        self.action_time = 0
        
    def draw_eyes(self, frame, eye_positions, eyes_closed, ear):
        if eye_positions is None:
            return frame
        
        for side, eye in eye_positions.items():
            x, y, w, h = eye['x'], eye['y'], eye['w'], eye['h']
            center = eye['center']
            
            if eyes_closed:
                color = COLORS['eye_closed']
                label = "FECHADO"
            else:
                color = COLORS['eye_open']
                label = "ABERTO"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, center, 5, color, -1)
        
        return frame
    
    def draw_ear_meter(self, frame, ear, eyes_closed):
        h, w = frame.shape[:2]
        
        meter_x = w - 60
        meter_y = 150
        meter_h = 300
        meter_w = 30
        
        cv2.rectangle(frame, (meter_x, meter_y), 
                     (meter_x + meter_w, meter_y + meter_h),
                     COLORS['panel'], -1)
        cv2.rectangle(frame, (meter_x, meter_y), 
                     (meter_x + meter_w, meter_y + meter_h),
                     COLORS['dim'], 2)
        
        level = np.clip(ear / 0.35, 0, 1)
        bar_h = int(meter_h * level)
        
        color = COLORS['eye_closed'] if eyes_closed else COLORS['eye_open']
        cv2.rectangle(frame, 
                     (meter_x + 3, meter_y + meter_h - bar_h),
                     (meter_x + meter_w - 3, meter_y + meter_h - 3),
                     color, -1)
        
        threshold_y = int(meter_y + meter_h * (1 - self.interpreter.eye_tracker.blink_threshold / 0.35))
        cv2.line(frame, (meter_x - 5, threshold_y), (meter_x + meter_w + 5, threshold_y),
                COLORS['warning'], 2)
        
        cv2.putText(frame, "EAR", (meter_x - 5, meter_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
        cv2.putText(frame, f"{ear:.2f}", (meter_x - 10, meter_y + meter_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def draw_particles(self, frame):
        for p in self.interpreter.particles:
            alpha = p['life']
            color = COLORS['dot'] if p['type'] == 'dot' else COLORS['dash']
            color = tuple(int(c * alpha) for c in color)
            size = int(6 * alpha)
            cv2.circle(frame, (int(p['x']), int(p['y'])), size, color, -1)
        return frame
    
    def draw_ui(self, frame, status, action, ear, eyes_closed):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (0, 0), (w, 90), COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        overlay = frame.copy()
        
        cv2.putText(frame, "MORSE", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['primary'], 2)
        cv2.putText(frame, "EYES", (170, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['secondary'], 2)
        
        cv2.putText(frame, "Comunique-se com piscadas!", (30, 78),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
        
        status_color = COLORS['eye_closed'] if eyes_closed else COLORS['eye_open']
        cv2.circle(frame, (w - 120, 45), 12, status_color, -1)
        cv2.putText(frame, status, (w - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        panel_y = 110
        panel_h = 130
        cv2.rectangle(overlay, (50, panel_y), (w - 100, panel_y + panel_h), COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        overlay = frame.copy()
        
        cv2.putText(frame, "CODIGO MORSE:", (70, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['dim'], 1)
        
        morse = self.interpreter.get_morse_code()
        x_offset = 70
        for i, symbol in enumerate(morse):
            if symbol == '.':
                cv2.circle(frame, (x_offset + i * 45, panel_y + 70), 14, COLORS['dot'], -1)
            else:
                cv2.rectangle(frame, (x_offset + i * 45 - 25, panel_y + 58),
                             (x_offset + i * 45 + 25, panel_y + 82), COLORS['dash'], -1)
        
        if not morse:
            cv2.putText(frame, "Pisque para inserir...", (70, panel_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['dim'], 1)
        
        if morse:
            predicted = MORSE_CODE.get(morse, '?')
            cv2.putText(frame, f"= {predicted}", (x_offset + len(morse) * 45 + 20, panel_y + 78),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['accent'], 2)
        
        word = self.interpreter.get_word()
        cv2.putText(frame, "PALAVRA:", (70, panel_y + 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
        cv2.putText(frame, word if word else "...", (180, panel_y + 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['text'], 2)
        
        sentence = self.interpreter.sentence
        if sentence or word:
            cv2.rectangle(overlay, (50, panel_y + panel_h + 20),
                         (w - 100, panel_y + panel_h + 80), COLORS['panel'], -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            overlay = frame.copy()
            
            display = sentence + word
            cv2.putText(frame, "MENSAGEM:", (70, panel_y + panel_h + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
            cv2.putText(frame, display[-45:], (70, panel_y + panel_h + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['success'], 2)
        
        if action:
            self.last_action = action
            self.action_time = time.time()
        
        if self.last_action and time.time() - self.action_time < 1.5:
            alpha = 1 - (time.time() - self.action_time) / 1.5
            cv2.putText(frame, self.last_action, (w // 2 - 80, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                       tuple(int(c * alpha) for c in COLORS['warning']), 2)
        
        inst_y = h - 180
        cv2.rectangle(overlay, (50, inst_y), (450, h - 20), COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        overlay = frame.copy()
        
        cv2.putText(frame, "COMO USAR:", (70, inst_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['primary'], 1)
        
        instructions = [
            ("Piscada rapida", "Ponto (.)", COLORS['dot']),
            ("Piscada longa", "Traco (-)", COLORS['dash']),
            ("2 piscadas", "Confirmar letra", COLORS['warning']),
            ("Fechar 2.5s", "Espaco/Palavra", COLORS['success']),
        ]
        
        for i, (action_desc, result, color) in enumerate(instructions):
            y = inst_y + 55 + i * 28
            cv2.putText(frame, action_desc, (70, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
            cv2.putText(frame, result, (230, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if self.show_reference:
            ref_x = w - 400
            cv2.rectangle(overlay, (ref_x, inst_y), (w - 100, h - 20), COLORS['panel'], -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            overlay = frame.copy()
            
            cv2.putText(frame, "TABELA MORSE:", (ref_x + 15, inst_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
            
            common = "ABCDEFGHIJKLMNOPQRST"
            for i, char in enumerate(common):
                col = i // 10
                row = i % 10
                x = ref_x + 15 + col * 145
                y = inst_y + 55 + row * 14
                morse_code = CHAR_TO_MORSE.get(char, '')
                cv2.putText(frame, f"{char}: {morse_code}", (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
        
        cv2.putText(frame, "[R] Toggle ref | [C] Limpar | [BACKSPACE] Apagar | [Q] Sair",
                   (w // 2 - 250, h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['dim'], 1)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not cap.isOpened():
            print("Erro: Nao foi possivel abrir a camera!")
            return
        
        print("\n" + "=" * 60)
        print("  MORSE EYES - Codigo Morse com Piscadas")
        print("=" * 60)
        print("\nComo usar:")
        print("  Piscada rapida (<0.25s)  = Ponto (.)")
        print("  Piscada longa (>0.4s)    = Traco (-)")
        print("  2 piscadas rapidas       = Confirmar letra")
        print("  Fechar olhos 2.5s        = Espaco/Confirmar palavra")
        print("\nDicas:")
        print("  - Mantenha o rosto bem iluminado")
        print("  - Olhe diretamente para a camera")
        print("  - Aguarde a calibracao inicial (~2s)")
        print("\nControles:")
        print("  [R]         - Toggle tabela morse")
        print("  [C]         - Limpar tudo")
        print("  [BACKSPACE] - Apagar ultimo simbolo")
        print("  [Q]         - Sair")
        print("=" * 60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            landmarks, eye_positions, action, ear, status = self.interpreter.process_frame(frame)
            eyes_closed = self.interpreter.eyes_closed
            
            if eye_positions:
                frame = self.draw_eyes(frame, eye_positions, eyes_closed, ear)
                frame = self.draw_ear_meter(frame, ear, eyes_closed)
            
            frame = self.draw_particles(frame)
            frame = self.draw_ui(frame, status, action, ear, eyes_closed)
            
            cv2.imshow('Morse Eyes', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.interpreter.clear_all()
                print("Limpo!")
            elif key == ord('r'):
                self.show_reference = not self.show_reference
            elif key == 8:
                self.interpreter.backspace()
        
        cap.release()
        cv2.destroyAllWindows()
        
        final = self.interpreter.sentence + self.interpreter.get_word()
        if final:
            print(f"\n{'=' * 60}")
            print(f"Mensagem final: {final}")
            print(f"{'=' * 60}\n")


if __name__ == "__main__":
    app = MorseEyesApp()
    app.run()
