import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# Paleta de cores cyberpunk/neon moderna
COLORS = {
    'bg': (15, 12, 25),
    'primary': (0, 255, 200),        # Ciano neon
    'secondary': (255, 100, 50),     # Laranja vibrante
    'accent': (255, 50, 150),        # Rosa neon
    'dot': (0, 255, 100),            # Verde neon para pontos
    'dash': (100, 150, 255),         # Azul claro para traços
    'text': (255, 255, 255),
    'dim': (120, 120, 140),
    'panel': (20, 18, 35),
    'panel_border': (60, 50, 100),
    'success': (50, 255, 150),
    'warning': (50, 200, 255),
    'danger': (80, 80, 255),         # Vermelho/azul para limpar
    'gold': (50, 200, 255),          # Dourado
    'glow': (180, 255, 100),         # Cor de brilho
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
    '.-...': '&', '---...': ':', '-.-.-.': ';', '-...-': '=',
    '.-.-.': '+', '-....-': '-', '..--.-': '_', '.-..-.': '"',
    '...-..-': '$', '.--.-.': '@',
}

CHAR_TO_MORSE = {v: k for k, v in MORSE_CODE.items()}

mp_hands = mp.solutions.hands


class MorseInterpreter:
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.8
        )
        
        self.current_gesture = None
        self.gesture_history = deque(maxlen=15)
        self.last_confirmed_gesture = None
        
        self.morse_buffer = []
        self.current_letter = ""
        self.word_buffer = []
        self.sentence = ""
        
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.8
        self.letter_timeout = 2.0
        self.last_input_time = 0
        self.gesture_released = True
        
        self.pulse_phase = 0
        self.particles = []
        
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
    
    def _detect_gesture(self, hand_landmarks, handedness="Right"):
        fingers = self._get_finger_states(hand_landmarks, handedness)
        thumb, index, middle, ring, pinky = fingers
        
        if fingers == [0, 0, 0, 0, 0]:
            return "DOT"
        
        if fingers == [1, 1, 1, 1, 1]:
            return "DASH"
        
        if fingers == [0, 1, 1, 0, 0]:
            return "SPACE"
        
        if fingers == [1, 0, 0, 0, 1]:
            return "WORD_SPACE"
        
        if fingers == [1, 0, 0, 0, 0]:
            return "BACKSPACE"
        
        # Novo gesto: 3 dedos (indicador, médio, anelar) = LIMPAR TUDO
        if fingers == [0, 1, 1, 1, 0]:
            return "CLEAR"
        
        return None
    
    def _morse_to_char(self, morse):
        return MORSE_CODE.get(morse, '?')
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        current_time = time.time()
        detected_gesture = None
        hand_landmarks = None
        action = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = "Right"
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            
            detected_gesture = self._detect_gesture(hand_landmarks, handedness)
            
            if detected_gesture:
                self.gesture_history.append(detected_gesture)
                
                if len(self.gesture_history) >= 10:
                    from collections import Counter
                    most_common = Counter(self.gesture_history).most_common(1)[0]
                    
                    if most_common[1] >= 8:
                        stable_gesture = most_common[0]
                        
                        can_input = False
                        if stable_gesture != self.last_confirmed_gesture:
                            if current_time - self.last_gesture_time > 0.4:
                                can_input = True
                        else:
                            if current_time - self.last_gesture_time > self.gesture_cooldown:
                                if self.gesture_released:
                                    can_input = True
                        
                        if can_input:
                            self.current_gesture = stable_gesture
                            self.last_confirmed_gesture = stable_gesture
                            self.last_gesture_time = current_time
                            self.last_input_time = current_time
                            self.gesture_released = False
                            
                            if stable_gesture == "DOT":
                                self.morse_buffer.append('.')
                                action = "PONTO (.)"
                                self._add_particle(frame.shape, "dot")
                            
                            elif stable_gesture == "DASH":
                                self.morse_buffer.append('-')
                                action = "TRACO (-)"
                                self._add_particle(frame.shape, "dash")
                            
                            elif stable_gesture == "SPACE":
                                if self.morse_buffer:
                                    morse = ''.join(self.morse_buffer)
                                    char = self._morse_to_char(morse)
                                    self.word_buffer.append(char)
                                    self.morse_buffer = []
                                    action = f"LETRA: {char}"
                            
                            elif stable_gesture == "WORD_SPACE":
                                if self.morse_buffer:
                                    morse = ''.join(self.morse_buffer)
                                    char = self._morse_to_char(morse)
                                    self.word_buffer.append(char)
                                    self.morse_buffer = []
                                
                                if self.word_buffer:
                                    word = ''.join(self.word_buffer)
                                    self.sentence += word + " "
                                    self.word_buffer = []
                                    action = f"PALAVRA: {word}"
                            
                            elif stable_gesture == "BACKSPACE":
                                if self.morse_buffer:
                                    self.morse_buffer.pop()
                                elif self.word_buffer:
                                    self.word_buffer.pop()
                                action = "APAGAR"
                            
                            elif stable_gesture == "CLEAR":
                                self.clear_all()
                                action = "LIMPAR TUDO"
                                self._add_particle(frame.shape, "clear")
            else:
                self.gesture_released = True
        else:
            self.current_gesture = None
            self.gesture_released = True
        
        if (self.morse_buffer and 
            current_time - self.last_input_time > self.letter_timeout):
            morse = ''.join(self.morse_buffer)
            char = self._morse_to_char(morse)
            self.word_buffer.append(char)
            self.morse_buffer = []
            action = f"AUTO-LETTER: {char}"
            self.last_input_time = current_time
        
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * np.pi)
        self._update_particles()
        
        return detected_gesture, hand_landmarks, action
    
    def _add_particle(self, shape, ptype):
        h, w = shape[:2]
        count = 15 if ptype == "clear" else 8
        for _ in range(count):
            self.particles.append({
                'x': w // 2 + np.random.randint(-100, 100),
                'y': h // 2 + np.random.randint(-100, 100),
                'vx': np.random.randn() * (8 if ptype == "clear" else 4),
                'vy': np.random.randn() * (8 if ptype == "clear" else 4),
                'life': 1.0,
                'type': ptype,
                'size': np.random.randint(5, 15)
            })
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 0.05
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


class MorseApp:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.interpreter = MorseInterpreter()
        self.show_reference = True
        self.last_action = None
        self.action_time = 0
        
    def draw_hand(self, frame, landmarks):
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
            cv2.line(frame, pt1, pt2, COLORS['primary'], 3)
        
        for i, lm in enumerate(landmarks.landmark):
            pt = (int(lm.x * w), int(lm.y * h))
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(frame, pt, 8, COLORS['secondary'], -1)
            else:
                cv2.circle(frame, pt, 4, COLORS['primary'], -1)
        
        return frame
    
    def draw_particles(self, frame):
        for p in self.interpreter.particles:
            alpha = p['life']
            if p['type'] == 'dot':
                color = COLORS['dot']
            elif p['type'] == 'dash':
                color = COLORS['dash']
            elif p['type'] == 'clear':
                color = COLORS['danger']
            else:
                color = COLORS['accent']
            color = tuple(int(c * alpha) for c in color)
            pt = (int(p['x']), int(p['y']))
            size = int(p.get('size', 5) * alpha)
            cv2.circle(frame, pt, size, color, -1)
            # Efeito de brilho
            if alpha > 0.5:
                cv2.circle(frame, pt, size + 3, color, 1)
        return frame
    
    def draw_panel(self, frame, x1, y1, x2, y2, alpha=0.8):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS['panel'], -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['panel_border'], 2)
    
    def draw_gradient_bar(self, frame, x1, y1, x2, y2, color1, color2):
        """Desenha uma barra com gradiente"""
        for i in range(x1, x2):
            ratio = (i - x1) / (x2 - x1)
            color = tuple(int(c1 + (c2 - c1) * ratio) for c1, c2 in zip(color1, color2))
            cv2.line(frame, (i, y1), (i, y2), color, 1)
    
    def draw_glow_text(self, frame, text, pos, font_scale, color, thickness=2):
        """Desenha texto com efeito de brilho"""
        # Glow effect
        glow_color = tuple(int(c * 0.3) for c in color)
        cv2.putText(frame, text, (pos[0]-2, pos[1]+2), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, glow_color, thickness + 4)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def draw_status_indicator(self, frame, x, y, active, label, color):
        """Desenha um indicador de status com LED"""
        # LED base
        cv2.circle(frame, (x, y), 8, COLORS['panel_border'], -1)
        if active:
            # LED ativo com glow
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 10, color, 1)
        else:
            cv2.circle(frame, (x, y), 4, COLORS['dim'], -1)
        # Label
        cv2.putText(frame, label, (x + 18, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color if active else COLORS['dim'], 1)
    
    def draw_ui(self, frame, gesture, action):
        h, w = frame.shape[:2]
        
        # ===== HEADER COM GRADIENTE =====
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), COLORS['panel'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Linha decorativa superior
        cv2.line(frame, (0, 88), (w, 88), COLORS['primary'], 2)
        
        # Título estilizado
        self.draw_glow_text(frame, "MORSE", (30, 58), 1.6, COLORS['primary'], 3)
        self.draw_glow_text(frame, "EYES", (200, 58), 1.6, COLORS['secondary'], 3)
        cv2.putText(frame, "Comunique-se com piscadas", (30, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['dim'], 1)
        
        # Status do gesto atual (canto direito)
        if gesture:
            gesture_colors = {
                'DOT': COLORS['dot'],
                'DASH': COLORS['dash'],
                'SPACE': COLORS['warning'],
                'WORD_SPACE': COLORS['success'],
                'BACKSPACE': COLORS['accent'],
                'CLEAR': COLORS['danger']
            }
            gesture_labels = {
                'DOT': 'PONTO',
                'DASH': 'TRACO', 
                'SPACE': 'CONFIRMAR',
                'WORD_SPACE': 'ESPACO',
                'BACKSPACE': 'APAGAR',
                'CLEAR': 'LIMPAR'
            }
            color = gesture_colors.get(gesture, COLORS['text'])
            label = gesture_labels.get(gesture, gesture)
            
            # Box do status
            box_w = 180
            cv2.rectangle(frame, (w - box_w - 20, 15), (w - 20, 75), color, 2)
            cv2.rectangle(frame, (w - box_w - 18, 17), (w - 22, 73), COLORS['panel'], -1)
            
            # LED indicador
            cv2.circle(frame, (w - box_w - 5, 45), 6, color, -1)
            cv2.circle(frame, (w - box_w - 5, 45), 9, color, 1)
            
            cv2.putText(frame, label, (w - box_w + 10, 52), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            # Status inativo
            self.draw_status_indicator(frame, w - 100, 45, False, "PRONTO", COLORS['success'])
        
        # ===== PAINEL PRINCIPAL - CÓDIGO MORSE =====
        panel_y = 105
        panel_h = 140
        self.draw_panel(frame, 50, panel_y, w - 50, panel_y + panel_h, 0.9)
        
        # Título do painel
        cv2.putText(frame, "CODIGO MORSE:", (70, panel_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['dim'], 1)
        
        morse = self.interpreter.get_morse_code()
        
        # Desenha símbolos morse com estilo melhorado
        x_offset = 85
        symbol_spacing = 55
        for i, symbol in enumerate(morse):
            x = x_offset + i * symbol_spacing
            if symbol == '.':
                # Ponto com glow
                cv2.circle(frame, (x, panel_y + 70), 18, COLORS['dot'], -1)
                cv2.circle(frame, (x, panel_y + 70), 22, COLORS['dot'], 2)
            else:
                # Traço com glow
                cv2.rectangle(frame, (x - 28, panel_y + 55), (x + 28, panel_y + 85), COLORS['dash'], -1)
                cv2.rectangle(frame, (x - 30, panel_y + 53), (x + 30, panel_y + 87), COLORS['dash'], 2)
        
        # Texto de espera animado
        if not morse:
            pulse = abs(np.sin(time.time() * 2)) * 0.5 + 0.5
            dim_color = tuple(int(c * pulse) for c in COLORS['dim'])
            cv2.putText(frame, "_ Aguardando input...", (70, panel_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, dim_color, 1)
        
        # Letra prevista (lado direito com destaque)
        if morse:
            predicted = MORSE_CODE.get(morse, '?')
            # Box da previsão
            pred_x = w - 150
            cv2.rectangle(frame, (pred_x - 30, panel_y + 30), (pred_x + 70, panel_y + 110), COLORS['accent'], 2)
            cv2.putText(frame, "=", (pred_x - 20, panel_y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['dim'], 2)
            self.draw_glow_text(frame, predicted, (pred_x + 10, panel_y + 95), 2.5, COLORS['accent'], 3)
        
        # Palavra atual
        word = self.interpreter.get_word()
        cv2.putText(frame, "PALAVRA:", (70, panel_y + 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
        display_word = word if word else "..."
        cv2.putText(frame, display_word, (180, panel_y + 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['primary'], 2)
        
        # ===== PAINEL DA MENSAGEM COMPLETA =====
        sentence = self.interpreter.sentence
        msg_panel_y = panel_y + panel_h + 15
        self.draw_panel(frame, 50, msg_panel_y, w - 50, msg_panel_y + 55, 0.9)
        
        cv2.putText(frame, "MENSAGEM:", (70, msg_panel_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['dim'], 1)
        
        display = sentence + word
        if display:
            # Mostra os últimos 45 caracteres
            self.draw_glow_text(frame, display[-45:], (70, msg_panel_y + 48), 0.85, COLORS['success'], 2)
        else:
            cv2.putText(frame, "Sua mensagem aparecera aqui...", (70, msg_panel_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['dim'], 1)
        
        # ===== AÇÃO ATUAL (FEEDBACK VISUAL) =====
        if action:
            self.last_action = action
            self.action_time = time.time()
        
        if self.last_action and time.time() - self.action_time < 1.2:
            alpha = 1 - (time.time() - self.action_time) / 1.2
            
            # Determina cor baseada na ação
            if "LIMPAR" in self.last_action:
                action_color = COLORS['danger']
            elif "PONTO" in self.last_action:
                action_color = COLORS['dot']
            elif "TRACO" in self.last_action:
                action_color = COLORS['dash']
            elif "LETRA" in self.last_action:
                action_color = COLORS['accent']
            else:
                action_color = COLORS['warning']
            
            color = tuple(int(c * alpha) for c in action_color)
            
            # Background semi-transparente para a ação
            overlay = frame.copy()
            text_size = cv2.getTextSize(self.last_action, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
            text_x = w // 2 - text_size[0] // 2
            text_y = h // 2 - 50
            
            cv2.rectangle(overlay, (text_x - 20, text_y - 40), 
                         (text_x + text_size[0] + 20, text_y + 20), COLORS['panel'], -1)
            cv2.addWeighted(overlay, 0.8 * alpha, frame, 1 - 0.8 * alpha, 0, frame)
            
            cv2.rectangle(frame, (text_x - 20, text_y - 40), 
                         (text_x + text_size[0] + 20, text_y + 20), color, 2)
            self.draw_glow_text(frame, self.last_action, (text_x, text_y), 1.3, color, 3)
        
        # ===== PAINEL DE GESTOS (ESQUERDA) =====
        ref_y = h - 200
        self.draw_panel(frame, 35, ref_y, 400, h - 20, 0.92)
        
        cv2.putText(frame, "COMO USAR:", (55, ref_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS['primary'], 2)
        
        # Linha separadora
        cv2.line(frame, (55, ref_y + 38), (380, ref_y + 38), COLORS['panel_border'], 1)
        
        gestures = [
            ("Punho fechar", "Ponto (.)", COLORS['dot']),
            ("Mao aberta", "Traco (-)", COLORS['dash']),
            ("Paz (V)", "Confirmar letra", COLORS['warning']),
            ("Hang loose", "Espaco/Palavra", COLORS['success']),
            ("Polegar", "Apagar ultimo", COLORS['accent']),
            ("3 dedos", "LIMPAR TUDO", COLORS['danger']),
        ]
        
        for i, (name, desc, color) in enumerate(gestures):
            y = ref_y + 58 + i * 24
            cv2.putText(frame, name, (55, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
            cv2.putText(frame, desc, (180, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # ===== TABELA MORSE (DIREITA) =====
        if self.show_reference:
            ref_x = w - 370
            self.draw_panel(frame, ref_x, ref_y, w - 35, h - 20, 0.92)
            
            cv2.putText(frame, "TABELA MORSE:", (ref_x + 15, ref_y + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['secondary'], 2)
            
            # Linha separadora
            cv2.line(frame, (ref_x + 15, ref_y + 38), (w - 50, ref_y + 38), COLORS['panel_border'], 1)
            
            common = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
            
            for i, char in enumerate(common):
                col = i // 10
                row = i % 10
                x = ref_x + 25 + col * 165
                y = ref_y + 58 + row * 14
                morse_code = CHAR_TO_MORSE.get(char, '')
                cv2.putText(frame, f"{char}: {morse_code}", (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS['text'], 1)
        
        # ===== BARRA DE ATALHOS (INFERIOR) =====
        shortcut_y = h - 8
        shortcuts = "[R] Toggle ref | [C] Limpar | [BACKSPACE] Apagar | [Q] Sair"
        cv2.putText(frame, shortcuts, (w // 2 - 220, shortcut_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['dim'], 1)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not cap.isOpened():
            print("Erro: Nao foi possivel abrir a camera!")
            return
        
        print("\n" + "═" * 60)
        print("   ╔╦╗╔═╗╦═╗╔═╗╔═╗  ╔═╗╦ ╦╔═╗╔═╗")
        print("   ║║║║ ║╠╦╝╚═╗║╣   ║╣ ╚╦╝║╣ ╚═╗")
        print("   ╩ ╩╚═╝╩╚═╚═╝╚═╝  ╚═╝ ╩ ╚═╝╚═╝")
        print("═" * 60)
        print("\n  🖐️  GESTOS:")
        print("  ═══════════════════════════════════════")
        print("  ✊ Punho fechado    →  Ponto (.)")
        print("  🖐️  Mao aberta      →  Traço (-)")
        print("  ✌️  Paz (V)         →  Confirmar letra")
        print("  🤙 Hang loose      →  Espaco/Palavra")
        print("  👍 Polegar         →  Apagar ultimo")
        print("  🤟 3 dedos         →  LIMPAR TUDO")
        print("\n  ⌨️  TECLADO:")
        print("  ═══════════════════════════════════════")
        print("  [R] Mostrar/ocultar tabela morse")
        print("  [C] Limpar tudo")
        print("  [BACKSPACE] Apagar ultimo")
        print("  [Q] Sair")
        print("═" * 60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            gesture, landmarks, action = self.interpreter.process_frame(frame)
            
            if landmarks:
                frame = self.draw_hand(frame, landmarks)
            
            frame = self.draw_particles(frame)
            frame = self.draw_ui(frame, gesture, action)
            
            cv2.imshow('Morse Code Interpreter', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.interpreter.clear_all()
                self.last_action = "LIMPAR TUDO"
                self.action_time = time.time()
                print("\n✨ Tudo limpo!")
            elif key == ord('r'):
                self.show_reference = not self.show_reference
            elif key == 8 or key == 127:  # Backspace
                if self.interpreter.morse_buffer:
                    self.interpreter.morse_buffer.pop()
                elif self.interpreter.word_buffer:
                    self.interpreter.word_buffer.pop()
                self.last_action = "APAGAR"
                self.action_time = time.time()
        
        cap.release()
        cv2.destroyAllWindows()
        
        final = self.interpreter.sentence + self.interpreter.get_word()
        if final:
            print(f"\n{'═' * 60}")
            print(f"  📝 MENSAGEM FINAL:")
            print(f"  {final}")
            print(f"{'═' * 60}\n")
        else:
            print("\n  👋 Ate a proxima!\n")


if __name__ == "__main__":
    app = MorseApp()
    app.run()
