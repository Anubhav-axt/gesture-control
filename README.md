# ✋ Gesture Control — Control Your Laptop with Hand Gestures

Control your Linux desktop with nothing but your webcam and hand gestures. No extra hardware needed.

---

## 🖐️ Gesture Map

| Gesture | Action |
|---|---|
| 👆 Pinch (index + thumb) | Left Click |
| ✌️ Two-finger pinch (middle + thumb) | Right Click |
| 🖐️ Wave hand Up / Down | Scroll Up / Down |
| 👋 Wave hand Left / Right | Switch Apps (Alt+Tab) |
| ✌️ Two fingers + wave Up / Down | Volume Up / Down |
| 🤌 Three fingers + wave Up / Down | Brightness Up / Down |
| ✋ Open palm → push toward screen | Play / Pause Media |
| 🌟 Open palm + fingers spread wide | Screenshot |

---

## 🛠️ Requirements

- Python 3.10+
- A webcam
- Linux (tested on Arch with Hyprland/i3wm)
- `brightnessctl` for brightness control

### Install dependencies

```bash
pip install mediapipe opencv-python pyautogui pynput
```

For brightness control:
```bash
# Arch Linux
sudo pacman -S brightnessctl
```

---

## 🚀 Run

```bash
python gesture_control.py
```

Press **Q** to quit.

---

## 📁 Project Structure

```
gesture-control/
├── gesture_control.py   # Main script
└── README.md
```

---

## 🔧 Customization

- **Scroll speed** → change the `3` in `pyautogui.scroll(3)`
- **Brightness step** → change `5%+` / `5%-` in the `brightnessctl` call
- **Sensitivity thresholds** → tweak `SWIPE_THRESH` and `SCROLL_THRESH` in the script
- **Screenshot shortcut** → update `pyautogui.hotkey("ctrl", "shift", "s")` to match your desktop environment

---

## 🧠 Tech Stack

| Tool | Purpose |
|---|---|
| [MediaPipe](https://mediapipe.dev/) | Hand landmark detection |
| [OpenCV](https://opencv.org/) | Webcam feed processing |
| [PyAutoGUI](https://pyautogui.readthedocs.io/) | Mouse & keyboard simulation |
| [pynput](https://pynput.readthedocs.io/) | Low-level keyboard control |

---

## 💡 Tips

- Make sure your hand is clearly visible and well-lit
- Keep your hand ~30–60cm from the webcam for best detection
- One hand only — multi-hand support coming soon!

---

## 📜 License

MIT
