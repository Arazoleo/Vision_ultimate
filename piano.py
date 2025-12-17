import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

try:
    import sounddevice as sd
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

COLORS = {
    'bg': (15, 12, 20), 'white_key': (250, 250, 255), 'white_key_pressed': (200, 220, 255),
    'black_key': (25, 25, 30), 'black_key_pressed': (60, 50, 80), 'primary': (255, 150, 100),
    'secondary': (150, 100, 255), 'accent': (100, 255, 200), 'glow': (255, 200, 150),
    'text': (255, 255, 255), 'dim': (100, 100, 120), 'panel': (25, 22, 35),
}

NOTES = {
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63,
    'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00,
    'A#4': 466.16, 'B4': 493.88, 'C5': 523.25, 'C#5': 554.37, 'D5': 587.33,
    'D#5': 622.25, 'E5': 659.25,
}

mp_hands = mp.solutions.hands


class SoundEngine:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def generate_tone(self, frequency, duration=0.3, volume=0.3):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * 0.6
        tone += np.sin(2 * np.pi * frequency * 2 * t) * 0.2
        tone += np.sin(2 * np.pi * frequency * 3 * t) * 0.1
        
        attack = int(0.01 * self.sample_rate)
        release = int(0.15 * self.sample_rate)
        envelope = np.ones(len(t))
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(0.7, 0, release)
        
        return (tone * envelope * volume).astype(np.float32)
    
    def play_note(self, note_name):
        if not SOUND_AVAILABLE or note_name not in NOTES:
            return
        try:
            tone = self.generate_tone(NOTES[note_name])
            sd.play(tone, self.sample_rate, blocking=False)
        except:
            pass


class PianoKey:
    def __init__(self, x, y, width, height, note, is_black=False):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.note, self.is_black = note, is_black
        self.pressed, self.press_time, self.glow_intensity = False, 0, 0
        
    def contains(self, px, py):
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height
    
    def press(self):
        if not self.pressed:
            self.pressed, self.press_time, self.glow_intensity = True, time.time(), 1.0
            return True
        return False
    
    def update(self):
        if self.glow_intensity > 0:
            self.glow_intensity -= 0.05
        if self.pressed and time.time() - self.press_time > 0.3:
            self.pressed = False
    
    def draw(self, frame):
        if self.is_black:
            base_color = COLORS['black_key_pressed'] if self.pressed else COLORS['black_key']
        else:
            base_color = COLORS['white_key_pressed'] if self.pressed else COLORS['white_key']
        
        if self.glow_intensity > 0 and not self.is_black:
            glow_color = tuple(int(c * self.glow_intensity) for c in COLORS['glow'])
            cv2.rectangle(frame, (self.x - 5, self.y - 5), 
                         (self.x + self.width + 5, self.y + self.height + 5), glow_color, -1)
        
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), base_color, -1)
        border_color = COLORS['accent'] if self.pressed else COLORS['dim']
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height), border_color, 2)
        
        if not self.is_black:
            note_display = self.note.replace('4', '').replace('5', '')
            text_color = COLORS['secondary'] if self.pressed else COLORS['dim']
            cv2.putText(frame, note_display, (self.x + self.width // 2 - 8, self.y + self.height - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


class Piano:
    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.keys = []
        self.sound_engine = SoundEngine()
        self.last_notes = deque(maxlen=20)
        self._create_keys()
        
    def _create_keys(self):
        white_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5']
        white_width = self.width // len(white_notes)
        
        for i, note in enumerate(white_notes):
            self.keys.append(PianoKey(self.x + i * white_width, self.y, white_width - 2, self.height, note))
        
        black_notes = ['C#4', 'D#4', None, 'F#4', 'G#4', 'A#4', None, 'C#5', 'D#5']
        black_width = int(white_width * 0.6)
        black_height = int(self.height * 0.6)
        
        for i, note in enumerate(black_notes):
            if note:
                self.keys.append(PianoKey(self.x + (i + 1) * white_width - black_width // 2,
                                         self.y, black_width, black_height, note, True))
    
    def check_press(self, finger_tips):
        played = []
        for key in self.keys:
            if key.is_black:
                for tip_x, tip_y in finger_tips:
                    if key.contains(tip_x, tip_y) and key.press():
                        self.sound_engine.play_note(key.note)
                        played.append(key.note)
                        self.last_notes.append((key.note, time.time()))
                        break
        
        for key in self.keys:
            if not key.is_black:
                for tip_x, tip_y in finger_tips:
                    if key.contains(tip_x, tip_y):
                        on_black = any(bk.is_black and bk.contains(tip_x, tip_y) for bk in self.keys)
                        if not on_black and key.press():
                            self.sound_engine.play_note(key.note)
                            played.append(key.note)
                            self.last_notes.append((key.note, time.time()))
                        break
        return played
    
    def update(self):
        for key in self.keys:
            key.update()
    
    def draw(self, frame):
        cv2.rectangle(frame, (self.x - 10, self.y - 10),
                     (self.x + self.width + 10, self.y + self.height + 20), COLORS['panel'], -1)
        for key in self.keys:
            if not key.is_black:
                key.draw(frame)
        for key in self.keys:
            if key.is_black:
                key.draw(frame)
        return frame


class AIPianoApp:
    def __init__(self):
        self.width, self.height = 1280, 720
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
        self.piano = Piano(50, self.height - 250, self.width - 100, 200)
        self.show_landmarks = True
        self.trail_points = deque(maxlen=50)
        self.particles = []
        
    def process_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        finger_tips, hand_landmarks_list = [], []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)
                h, w = frame.shape[:2]
                for tip_id in [8, 12, 16, 20, 4]:
                    tip = hand_landmarks.landmark[tip_id]
                    tip_x, tip_y = int(tip.x * w), int(tip.y * h)
                    finger_tips.append((tip_x, tip_y))
                    self.trail_points.append((tip_x, tip_y, time.time()))
        return finger_tips, hand_landmarks_list
    
    def draw_hands(self, frame, hand_landmarks_list):
        if not self.show_landmarks:
            return frame
        h, w = frame.shape[:2]
        connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
                      (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        
        for hand_landmarks in hand_landmarks_list:
            for start, end in connections:
                pt1 = (int(hand_landmarks.landmark[start].x * w), int(hand_landmarks.landmark[start].y * h))
                pt2 = (int(hand_landmarks.landmark[end].x * w), int(hand_landmarks.landmark[end].y * h))
                cv2.line(frame, pt1, pt2, COLORS['primary'], 2)
            
            for tip_id in [4, 8, 12, 16, 20]:
                tip = hand_landmarks.landmark[tip_id]
                pt = (int(tip.x * w), int(tip.y * h))
                cv2.circle(frame, pt, 18, COLORS['glow'], -1)
                cv2.circle(frame, pt, 12, COLORS['accent'], -1)
                cv2.circle(frame, pt, 6, COLORS['text'], -1)
        return frame
    
    def draw_trail(self, frame):
        current_time = time.time()
        for x, y, t in self.trail_points:
            if current_time - t < 0.5:
                alpha = 1 - (current_time - t) / 0.5
                cv2.circle(frame, (x, y), int(8 * alpha), tuple(int(c * alpha) for c in COLORS['accent']), -1)
        return frame
    
    def add_particles(self, x, y, note):
        for _ in range(10):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, 8)
            self.particles.append({'x': x, 'y': y, 'vx': np.cos(angle) * speed,
                                  'vy': np.sin(angle) * speed - 3, 'life': 1.0,
                                  'color': COLORS['accent'] if '#' not in note else COLORS['secondary']})
    
    def update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.2
            p['life'] -= 0.03
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def draw_particles(self, frame):
        for p in self.particles:
            color = tuple(int(c * p['life']) for c in p['color'])
            cv2.circle(frame, (int(p['x']), int(p['y'])), int(5 * p['life']), color, -1)
        return frame
    
    def draw_ui(self, frame, played_notes):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        overlay = frame.copy()
        
        cv2.putText(frame, "AI PIANO", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['primary'], 2)
        
        if SOUND_AVAILABLE:
            cv2.circle(frame, (w - 30, 35), 8, COLORS['accent'], -1)
            cv2.putText(frame, "SOM", (w - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['accent'], 1)
        else:
            cv2.circle(frame, (w - 30, 35), 8, COLORS['dim'], -1)
            cv2.putText(frame, "MUDO", (w - 90, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
        
        recent = list(self.piano.last_notes)[-8:]
        if recent:
            cv2.putText(frame, f"Notas: {' '.join([n for n, t in recent])}", (w // 2 - 150, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['accent'], 2)
        
        if played_notes:
            cv2.putText(frame, " | ".join(played_notes), (w // 2 - 50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['glow'], 2)
        
        cv2.putText(frame, "Use os dedos para tocar as teclas!", (30, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['dim'], 1)
        cv2.putText(frame, "[L] Toggle maos | [Q] Sair", (w - 280, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
        
        cv2.line(frame, (50, self.piano.y - 20), (w - 50, self.piano.y - 20), COLORS['dim'], 1)
        cv2.putText(frame, "ZONA DE TOQUE", (w // 2 - 70, self.piano.y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not cap.isOpened():
            print("Erro: Nao foi possivel abrir a camera!")
            return
        
        print("\n" + "=" * 60)
        print("  AI PIANO - Toque com os dedos no ar!")
        print("=" * 60)
        print(f"\nSom: {'ATIVADO' if SOUND_AVAILABLE else 'DESATIVADO'}")
        print("\nControles: [L] Toggle maos | [Q] Sair")
        print("=" * 60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            finger_tips, hand_landmarks = self.process_hands(frame)
            played = self.piano.check_press(finger_tips)
            
            for note in played:
                for key in self.piano.keys:
                    if key.note == note:
                        self.add_particles(key.x + key.width // 2, key.y + key.height // 2, note)
            
            self.piano.update()
            self.update_particles()
            
            frame = self.draw_trail(frame)
            frame = self.draw_hands(frame, hand_landmarks)
            frame = self.piano.draw(frame)
            frame = self.draw_particles(frame)
            frame = self.draw_ui(frame, played)
            
            cv2.imshow('AI Piano', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSessao encerrada! Notas tocadas: {len(list(self.piano.last_notes))}\n")


if __name__ == "__main__":
    AIPianoApp().run()
