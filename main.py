import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
from collections import deque

class MirrorCloneFX:
    def __init__(self,
                 camera_index=0,
                 window_width=1280,
                 window_height=720,
                 max_particles=200,
                 gesture_history_len=6,
                 gesture_switch_cooldown=0.6,
                 min_detection_confidence=0.6):
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Visual modes
        self.modes = {
            0: "Dots",
            1: "Lines",
            2: "ASCII",
            3: "Particles"
        }
        self.current_mode = 0

        # Particles system
        self.particles = []
        self.max_particles = max_particles

        # ASCII characters for ASCII mode (dense -> sparse)
        self.ascii_chars = "█▉▊▋▌▍▎▏ "

        # Window / camera
        self.window_width = window_width
        self.window_height = window_height
        self.half_width = self.window_width // 2
        self.camera_index = camera_index

        # Gesture smoothing / debounce
        self.gesture_history = deque(maxlen=gesture_history_len)
        self.last_switch_time = 0.0
        self.switch_cooldown = gesture_switch_cooldown

        # Angle thresholds (degrees) - tweak these if needed
        self.extended_finger_angle_thresh = 150  # angle > this -> finger considered extended
        self.curled_finger_angle_thresh = 100    # angle < this -> finger considered curled

    # -------------------------
    # Geometry helpers
    # -------------------------
    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculate angle ABC (in degrees) where a,b,c are landmarks with x,y coords.
        """
        ax, ay = a.x, a.y
        bx, by = b.x, b.y
        cx, cy = c.x, c.y

        # vectors BA and BC
        v1 = np.array([ax - bx, ay - by])
        v2 = np.array([cx - bx, cy - by])

        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if mag1 == 0 or mag2 == 0:
            return 0.0

        # clamp dot product to avoid numerical errors
        cos_angle = np.dot(v1, v2) / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle

    # -------------------------
    # Gesture detection
    # -------------------------
    def detect_fingers_from_landmarks(self, landmarks):
        """
        Given MediaPipe landmarks (21), return booleans for thumb, index, middle, ring, pinky
        using angle-based approach. Returns tuple (thumb, index, middle, ring, pinky)
        where each is True if considered extended.
        """
        if not landmarks or len(landmarks) < 21:
            return (False, False, False, False, False)

        lm = landmarks

        # For each finger, measure the angle at the PIP/MCP joint.
        # Index: angle between landmarks 6 (PIP), 7 (DIP), 8 (TIP) or use 5-6-8
        # We'll use slightly different triplets that tend to be stable:
        angle_index = self.calculate_angle(lm[5], lm[6], lm[8])   # MCP-PIP-TIP style
        angle_middle = self.calculate_angle(lm[9], lm[10], lm[12])
        angle_ring = self.calculate_angle(lm[13], lm[14], lm[16])
        angle_pinky = self.calculate_angle(lm[17], lm[18], lm[20])

        # Thumb: measure angle between landmarks 2, 3, 4 (base -> mid -> tip)
        # Also consider relative x-distance to handle left/right
        angle_thumb = self.calculate_angle(lm[1], lm[2], lm[4])

        # Determine extended / curled using thresholds
        thumb_up = angle_thumb > self.extended_finger_angle_thresh
        index_up = angle_index > self.extended_finger_angle_thresh
        middle_up = angle_middle > self.extended_finger_angle_thresh
        ring_up = angle_ring > self.extended_finger_angle_thresh
        pinky_up = angle_pinky > self.extended_finger_angle_thresh

        # If angles are ambiguous, allow a "soft" threshold fallback:
        # e.g., if angle is near boundary, check relative tip distances.
        def tip_above_pip(tip_idx, pip_idx):
            return lm[tip_idx].y < lm[pip_idx].y

        # Soft corrections: if angle roughly 120-150 and tip clearly above pip, count as up
        for angle_val, tip_idx, pip_idx, name in [
            (angle_index, 8, 6, 'index'),
            (angle_middle, 12, 10, 'middle'),
            (angle_ring, 16, 14, 'ring'),
            (angle_pinky, 20, 18, 'pinky')
        ]:
            if self.curled_finger_angle_thresh < angle_val < self.extended_finger_angle_thresh:
                # soft decision by position
                if tip_above_pip(tip_idx, pip_idx):
                    if name == 'index':
                        index_up = True
                    elif name == 'middle':
                        middle_up = True
                    elif name == 'ring':
                        ring_up = True
                    elif name == 'pinky':
                        pinky_up = True

        # Return booleans in the original order (thumb, index, middle, ring, pinky)
        return (thumb_up, index_up, middle_up, ring_up, pinky_up)

    def detect_hand_gesture(self, results):
        """
        Wrapper that uses MediaPipe 'results' object.
        Returns gesture id (0..3) or None.
        Gesture mapping:
          0 -> Dots (two fingers: index+middle)
          1 -> Lines (one finger: index)
          2 -> ASCII (thumb + pinky / 'shaka')
          3 -> Particles (open palm: >=4 fingers)
        Uses smoothing (history) and cooldown.
        """
        # If results is None or no hand landmarks, push None into history and return None
        if not results or not results.multi_hand_landmarks:
            self.gesture_history.append(None)
            # choose stable gesture if buffer majority indicates something else
            return self._decide_gesture_from_history()

        # Check detection confidence (handedness)
        if results.multi_handedness and len(results.multi_handedness) > 0:
            score = results.multi_handedness[0].classification[0].score
            if score < 0.5:
                # Too low confidence; ignore
                self.gesture_history.append(None)
                return self._decide_gesture_from_history()

        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        fingers = self.detect_fingers_from_landmarks(hand_landmarks.landmark)  # tuple of bools

        # Convert bools -> list of ints like original code for compatibility
        fingers_list = [1 if x else 0 for x in fingers]  # [thumb, index, middle, ring, pinky]

        # Map finger pattern -> gesture id
        gesture = None
        # Two fingers index+middle -> Dots (0)
        if fingers_list == [0, 1, 1, 0, 0] or (fingers_list[1] and fingers_list[2] and not fingers_list[0]):
            gesture = 0
        # One finger (only index) -> Lines (1)
        elif fingers_list == [0, 1, 0, 0, 0] or (fingers_list[1] and not any([fingers_list[0], fingers_list[2], fingers_list[3], fingers_list[4]])):
            gesture = 1
        # Thumb + pinky (shaka) -> ASCII (2)
        elif fingers_list[0] and fingers_list[4] and not any([fingers_list[1], fingers_list[2], fingers_list[3]]):
            gesture = 2
        # Open palm -> Particles (3) (4 or 5 fingers)
        elif sum(fingers_list) >= 4:
            gesture = 3
        else:
            # fallback: if index is up and thumb slightly up then maybe lines etc.
            gesture = None

        # Append to history then decide stable gesture
        self.gesture_history.append(gesture)
        return self._decide_gesture_from_history()

    def _decide_gesture_from_history(self):
        """
        Decide the current gesture using the history buffer (majority vote).
        Also apply cooldown so gestures don't switch too often.
        """
        # If no history yet, return None
        if len(self.gesture_history) == 0:
            return None

        # Count occurrences ignoring None
        votes = [g for g in self.gesture_history if g is not None]
        if not votes:
            return None

        # majority vote
        vote = max(set(votes), key=votes.count)

        # apply cooldown before switching
        now = time.time()
        if vote != self.current_mode and now - self.last_switch_time > self.switch_cooldown:
            # allow switch
            self.last_switch_time = now
            return vote
        else:
            # keep previous mode
            return self.current_mode

    # -------------------------
    # Visual effects
    # -------------------------
    def create_dots_effect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.zeros_like(frame)
        height, width = gray.shape
        dot_spacing = 12

        for y in range(0, height, dot_spacing):
            for x in range(0, width, dot_spacing):
                intensity = int(gray[y, x])
                if intensity > 60:
                    radius = int((intensity / 255) * 6) + 1
                    color = frame[y, x].astype(np.float32)
                    color = np.clip(color * 1.2, 0, 255).astype(np.uint8)
                    cv2.circle(result, (x, y), radius, color.tolist(), -1)
        return result

    def create_lines_effect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 80)
        result = np.zeros_like(frame)
        ys, xs = np.where(edges > 0)
        for y, x in zip(ys, xs):
            original_color = frame[y, x].astype(np.float32)
            enhanced_color = np.clip(original_color * 1.5, 0, 255).astype(np.uint8)
            result[y, x] = enhanced_color
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1)
        return result

    def create_ascii_effect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.zeros_like(frame)
        height, width = gray.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58
        thickness = 1
        char_w = 14
        char_h = 18

        for y in range(0, height, char_h):
            for x in range(0, width, char_w):
                if y + char_h < height and x + char_w < width:
                    region = gray[y:y + char_h, x:x + char_w]
                    avg = np.mean(region)
                    if avg > 25:
                        char_index = int(((255 - avg) / 255) * (len(self.ascii_chars) - 1))
                        char = self.ascii_chars[char_index]
                        if avg > 150:
                            color = (255, 255, 255)
                        elif avg > 100:
                            color = (0, 255, 0)
                        else:
                            color = (0, 255, 255)
                        cv2.putText(result, char, (x + 2, y + char_h - 4),
                                    font, font_scale, color, thickness, cv2.LINE_AA)
        return result

    def update_particles(self, frame, landmarks):
        if landmarks:
            for i in range(0, len(landmarks), 2):  # every other landmark
                if len(self.particles) >= self.max_particles:
                    break
                lm = landmarks[i]
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                particle = {
                    'x': x + random.randint(-18, 18),
                    'y': y + random.randint(-18, 18),
                    'vx': random.uniform(-2.0, 2.0),
                    'vy': random.uniform(-2.0, 2.0),
                    'life': random.randint(40, 80),
                    'color': [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
                }
                self.particles.append(particle)

        # Age particles and remove dead ones
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.08  # gravity
            p['life'] -= 1

    def create_particles_effect(self, frame, landmarks):
        result = np.zeros_like(frame)
        self.update_particles(frame, landmarks)
        for particle in self.particles:
            x, y = int(particle['x']), int(particle['y'])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                alpha = particle['life'] / 80.0
                radius = max(1, int(alpha * 5))
                color = [int(c * alpha) for c in particle['color']]
                cv2.circle(result, (x, y), radius, color, -1)
        return result

    # -------------------------
    # Frame processing / main loop
    # -------------------------
    def process_frame(self, frame, results):
        # Use current_mode to produce stylized output; pass landmarks too for particles
        landmarks = None
        if results and results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark

        if self.current_mode == 0:
            return self.create_dots_effect(frame)
        elif self.current_mode == 1:
            return self.create_lines_effect(frame)
        elif self.current_mode == 2:
            return self.create_ascii_effect(frame)
        elif self.current_mode == 3:
            return self.create_particles_effect(frame, landmarks)
        else:
            return frame

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Try to set desired resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("MirrorCloneFX started.")
        print("Gestures:")
        print("  ✌️  Two fingers (index+middle) → Dots mode")
        print("  ☝️  One finger (index) → Lines mode")
        print("  🤙 Thumb + pinky → ASCII mode")
        print("  ✋ Open palm (4+ fingers) → Particles mode")
        print("Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame.")
                    break

                frame = cv2.flip(frame, 1)  # mirror
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe processing
                results = self.hands.process(frame_rgb)

                # Detect gesture and set mode (detect_hand_gesture handles smoothing + cooldown)
                gesture = self.detect_hand_gesture(results)
                if gesture is not None and gesture != self.current_mode:
                    # Only update current_mode when detect_hand_gesture's decision allows switching
                    # note: detect_hand_gesture already respects cooldown; we update here to reflect final decision
                    self.current_mode = gesture

                # Draw hand landmarks on original left panel for feedback
                display_frame = frame.copy()
                if results and results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(display_frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)

                # Compose split-screen (original | stylized)
                # Resize left and right halves to ensure consistent layout
                frame_resized = cv2.resize(display_frame, (self.half_width, self.window_height))
                stylized = self.process_frame(frame, results)
                stylized_resized = cv2.resize(stylized, (self.half_width, self.window_height))
                split = np.hstack((frame_resized, stylized_resized))

                # Mode label + divider
                mode_text = f"Mode: {self.modes.get(self.current_mode, 'Unknown')}"
                cv2.putText(split, mode_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(split, (self.half_width, 0), (self.half_width, self.window_height), (255, 255, 255), 2)
                cv2.putText(split, "Original", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(split, "Clone", (self.half_width + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("MirrorCloneFX", split)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    app = MirrorCloneFX()
    app.run()

if __name__ == "__main__":
    main()
