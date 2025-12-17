# Vision AI Ultimate

Sistema de visão computacional em tempo real com IA.

## Features Principais

- **Emotion AI** - Detecção de emoções faciais (feliz, triste, raiva, surpresa)
- **Object Detection** - YOLO v8 para detectar objetos
- **Air Canvas** - Desenhe no ar com gestos
- **Pose Tracking** - Rastreamento do corpo
- **Hand Tracking** - Rastreamento de mãos e gestos
- **Focus Monitor** - Monitor de atenção
- **Fitness Tracker** - Contador de exercícios
- **Visual Effects** - Matrix, Glitch, Neon, Clone

---

## 🆕 Novos Módulos

### 🤟 Interpretador de Libras (`libras.py`)
Reconhece o alfabeto manual brasileiro de Libras usando visão computacional.

```bash
python libras.py
```

**Funcionalidades:**
- Reconhecimento de letras A-Z do alfabeto manual
- Formação de palavras e frases
- Feedback visual em tempo real
- Histórico de letras confirmadas

**Controles:**
| Tecla | Ação |
|-------|------|
| `SPACE` | Confirmar palavra |
| `BACKSPACE` | Apagar última letra |
| `C` | Limpar tudo |
| `L` | Toggle landmarks |
| `Q` | Sair |

---

### 📡 Interpretador de Código Morse (`morse.py`)
Transmita mensagens em código Morse usando gestos das mãos.

```bash
python morse.py
```

**Gestos:**
| Gesto | Símbolo |
|-------|---------|
| Punho fechado | Ponto (.) |
| Mão aberta | Traço (-) |
| Paz (V) | Confirmar letra |
| Hang loose | Espaço entre palavras |
| Polegar | Apagar |

**Controles:**
| Tecla | Ação |
|-------|------|
| `R` | Toggle tabela Morse |
| `C` | Limpar tudo |
| `Q` | Sair |

---

### 👁️ Morse Eyes (`morse_eyes.py`)
Código Morse usando **piscadas dos olhos** - perfeito para acessibilidade!

```bash
python morse_eyes.py
```

**Como funciona:**
| Ação | Significado |
|------|-------------|
| Piscada rápida (<0.25s) | Ponto (.) |
| Piscada longa (>0.4s) | Traço (-) |
| 2 piscadas rápidas | Confirmar letra |
| Fechar olhos 2.5s | Espaço entre palavras |

**Controles:**
| Tecla | Ação |
|-------|------|
| `R` | Toggle tabela Morse |
| `C` | Limpar tudo |
| `BACKSPACE` | Apagar último símbolo |
| `Q` | Sair |

---

### 🎹 AI Piano (`piano.py`)
Toque piano no ar usando os dedos!

```bash
python piano.py
```

**Funcionalidades:**
- 10 teclas brancas + 9 teclas pretas (2 oitavas)
- Som sintetizado em tempo real (requer `sounddevice`)
- Suporte para duas mãos
- Efeitos visuais de partículas
- Rastro dos dedos

**Controles:**
| Tecla | Ação |
|-------|------|
| `L` | Toggle landmarks das mãos |
| `Q` | Sair |

---

## Instalação

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Para som no AI Piano (opcional)
pip install sounddevice
```

## Executar

```bash
# Módulo principal (todos os modos)
python main.py

# Módulos individuais
python libras.py   # Interpretador de Libras
python morse.py    # Código Morse
python piano.py    # AI Piano
```

## Controles do Módulo Principal

| Tecla | Ação |
|-------|------|
| `1` | Modo Detection |
| `2` | Modo Emotion |
| `3` | Modo Canvas |
| `4` | Modo Effects |
| `5` | Modo Focus |
| `6` | Modo Fitness |
| `E` | Trocar efeito visual |
| `C` | Limpar canvas |
| `O` | Toggle objetos |
| `P` | Toggle pose |
| `H` | Toggle mãos |
| `F` | Toggle face |
| `Q` | Sair |

## Requisitos

- Python 3.9+
- Webcam
- OpenCV, MediaPipe, Ultralytics (YOLO)
- sounddevice (opcional, para AI Piano)

## Estrutura do Projeto

```
vision/
├── main.py          # Módulo principal com todos os modos
├── libras.py        # Interpretador de Libras
├── morse.py         # Código Morse (gestos)
├── morse_eyes.py    # Código Morse (piscadas) 👁️
├── piano.py         # AI Piano
├── requirements.txt # Dependências
├── yolov8n.pt       # Modelo YOLO
└── README.md        # Este arquivo
```
