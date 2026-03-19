import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui
import time
import math
import subprocess
from pynput.keyboard import Controller as KeyboardController

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
pyautogui.FAILSAFE = False
keyboard = KeyboardController()

# ── Cooldown tracker ──────────────────────────────────────────────────────────
last_action: dict[str, float] = {}

def cooldown(name: str, seconds: float = 0.6) -> bool:
    now = time.time()
    if now - last_action.get(name, 0) >= seconds:
        last_action[name] = now
        return True
    return False

# ── Geometry helpers ──────────────────────────────────────────────────────────
def dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4
INDEX_TIP  = 8
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20
INDEX_MCP  = 5
MIDDLE_MCP = 9
RING_MCP   = 13
PINKY_MCP  = 17

# ── Gesture detectors ─────────────────────────────────────────────────────────
def is_pinch(lm, threshold=0.06) -> bool:
    return dist(lm[THUMB_TIP], lm[INDEX_TIP]) < threshold

def is_two_finger_pinch(lm, threshold=0.06) -> bool:
    return dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) < threshold

def fingers_up(lm) -> list[bool]:
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return [lm[t].y < lm[m].y for t, m in zip(tips, mcps)]

def is_open_palm(lm) -> bool:
    return all(fingers_up(lm))

def is_two_fingers_up(lm) -> bool:
    up = fingers_up(lm)
    return up[0] and up[1] and not up[2] and not up[3]

def is_three_fingers_up(lm) -> bool:
    up = fingers_up(lm)
    return up[0] and up[1] and up[2] and not up[3]

# ── Motion tracker ────────────────────────────────────────────────────────────
class MotionTracker:
    WINDOW = 12

    def __init__(self):
        self.history: list[tuple[float, float]] = []

    def update(self, x: float, y: float):
        self.history.append((x, y))
        if len(self.history) > self.WINDOW:
            self.history.pop(0)

    def delta(self) -> tuple[float, float]:
        if len(self.history) < 4:
            return 0.0, 0.0
        dx = self.history[-1][0] - self.history[0][0]
        dy = self.history[-1][1] - self.history[0][1]
        return dx, dy

    def clear(self):
        self.history.clear()

class PushTracker:
    WINDOW = 10

    def __init__(self):
        self.z_history: list[float] = []

    def update(self, z: float):
        self.z_history.append(z)
        if len(self.z_history) > self.WINDOW:
            self.z_history.pop(0)

    def pushed(self, threshold=0.04) -> bool:
        if len(self.z_history) < self.WINDOW:
            return False
        return (self.z_history[0] - self.z_history[-1]) > threshold

    def clear(self):
        self.z_history.clear()

# ── HUD ───────────────────────────────────────────────────────────────────────
def show_hud(frame, gesture: str):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Gesture: {gesture}", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
    legend = [
        "Pinch          = Left Click",
        "2-finger pinch = Right Click",
        "Wave U/D       = Scroll",
        "Wave L/R       = Switch App",
        "2 fingers U/D  = Volume",
        "3 fingers U/D  = Brightness",
        "Open palm push = Play/Pause",
        "Star spread    = Screenshot",
    ]
    for i, line in enumerate(legend):
        cv2.putText(frame, line, (w - 330, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

# ── Draw landmarks ────────────────────────────────────────────────────────────
def draw_landmarks(frame, lm_list):
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(2)  # give camera time to warm up
    if not cap.isOpened():
        print("❌  Could not open webcam.")
        return

    print("📷  Webcam opened successfully!")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.75,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    motion       = MotionTracker()
    push_tracker = PushTracker()

    print("✅  Gesture Control running — press Q to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("⚠️  Failed to read frame.")
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        current_gesture = "—"

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_landmarks(frame, lm)

            wrist_x = lm[WRIST].x
            wrist_y = lm[WRIST].y
            wrist_z = lm[WRIST].z

            motion.update(wrist_x, wrist_y)
            dx, dy = motion.delta()

            if is_open_palm(lm) and dist(lm[PINKY_TIP], lm[INDEX_TIP]) <= 0.35:
                push_tracker.update(wrist_z)
                if push_tracker.pushed() and cooldown("playpause", 1.0):
                    pyautogui.press("playpause")
                    current_gesture = "▶/⏸  Play/Pause"
                    push_tracker.clear()
                else:
                    current_gesture = "✋ Open Palm (push for Play/Pause)"

            elif is_open_palm(lm) and dist(lm[PINKY_TIP], lm[INDEX_TIP]) > 0.35:
                if cooldown("screenshot", 1.5):
                    pyautogui.hotkey("ctrl", "shift", "s")
                    current_gesture = "📸 Screenshot"
                else:
                    current_gesture = "🌟 Star (Screenshot)"
                push_tracker.clear()

            elif is_pinch(lm):
                if cooldown("leftclick", 0.5):
                    pyautogui.click()
                    current_gesture = "👆 Left Click"

            elif is_two_finger_pinch(lm):
                if cooldown("rightclick", 0.5):
                    pyautogui.rightClick()
                    current_gesture = "✌️ Right Click"

            elif is_two_fingers_up(lm):
                current_gesture = "✌️ Volume mode"
                if abs(dy) > 0.08:
                    if dy < 0 and cooldown("volup", 0.3):
                        pyautogui.press("volumeup")
                        current_gesture = "🔊 Volume Up"
                        motion.clear()
                    elif dy > 0 and cooldown("voldown", 0.3):
                        pyautogui.press("volumedown")
                        current_gesture = "🔉 Volume Down"
                        motion.clear()

            elif is_three_fingers_up(lm):
                current_gesture = "☀️ Brightness mode"
                if abs(dy) > 0.08:
                    if dy < 0 and cooldown("brup", 0.3):
                        subprocess.Popen(["brightnessctl", "set", "5%+"])
                        current_gesture = "🌕 Brightness Up"
                        motion.clear()
                    elif dy > 0 and cooldown("brdown", 0.3):
                        subprocess.Popen(["brightnessctl", "set", "5%-"])
                        current_gesture = "🌑 Brightness Down"
                        motion.clear()

            else:
                push_tracker.clear()
                SWIPE_THRESH  = 0.18
                SCROLL_THRESH = 0.10

                if abs(dx) > SWIPE_THRESH:
                    if dx > 0 and cooldown("swiperight", 0.7):
                        pyautogui.hotkey("alt", "tab")
                        current_gesture = "👉 Switch App →"
                        motion.clear()
                    elif dx < 0 and cooldown("swipeleft", 0.7):
                        pyautogui.hotkey("alt", "shift", "tab")
                        current_gesture = "👈 Switch App ←"
                        motion.clear()
                elif abs(dy) > SCROLL_THRESH:
                    if dy < 0 and cooldown("scrollup", 0.15):
                        pyautogui.scroll(3)
                        current_gesture = "⬆️ Scroll Up"
                    elif dy > 0 and cooldown("scrolldown", 0.15):
                        pyautogui.scroll(-3)
                        current_gesture = "⬇️ Scroll Down"
        else:
            motion.clear()
            push_tracker.clear()

        show_hud(frame, current_gesture)
        cv2.imshow("✋ Gesture Control — press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()