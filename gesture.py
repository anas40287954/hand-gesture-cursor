import time
import numpy as np
import cv2
import mediapipe as mp
import pyautogui


class GestureController:
    """Real-time webcam computer vision interface for desktop mouse control."""

    def __init__(self, width: int = 640, height: int = 480):
        # Camera & Display Properties
        self.width = width
        self.height = height
        self.frame_reduction = 180  # Active Region of Interest (ROI) margin

        # Pointer Interpolation & Sensitivity
        self.smoothing = 5.0
        self.ploc_x, self.ploc_y = 0.0, 0.0
        self.cloc_x, self.cloc_y = 0.0, 0.0

        # Gesture Thresholds & Timers
        self.pinch_ratio_threshold = 0.22
        self.click_cooldown = 0.25
        self.last_click_time = 0.0
        self.pinch_start_time = 0.0
        self.pinch_start_pos = (0.0, 0.0)

        # Operational States
        self.is_pinched = False
        self.is_dragging = False

        # System Screen Constraints
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = True

        # MediaPipe Initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def _get_normalized_pinch_distance(p1, p2, wrist, middle_mcp, w: int, h: int) -> float:
        """Calculates distance between landmarks normalized by hand scale."""
        x1, y1 = p1.x * w, p1.y * h
        x2, y2 = p2.x * w, p2.y * h
        wx, wy = wrist.x * w, wrist.y * h
        mx, my = middle_mcp.x * w, middle_mcp.y * h

        pinch_dist = np.hypot(x2 - x1, y2 - y1)
        hand_scale = np.hypot(mx - wx, my - wy)

        return pinch_dist / hand_scale if hand_scale > 0 else 1.0

    @staticmethod
    def _evaluate_finger_states(landmarks, handedness: str) -> list[int]:
        """Evaluates extended/flexed states for all 5 fingers."""
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        fingers = []

        # Thumb logic based on camera orientation
        if handedness == "Right":
            fingers.append(1 if landmarks[tips[0]].x < landmarks[pips[0]].x else 0)
        else:
            fingers.append(1 if landmarks[tips[0]].x > landmarks[pips[0]].x else 0)

        # Remaining four fingers vertical check
        for i in range(1, 5):
            fingers.append(1 if landmarks[tips[i]].y < landmarks[pips[i]].y else 0)

        return fingers

    def _process_gestures(self, landmarks, handedness: str, current_time: float) -> str:
        """Evaluates geometric configurations and executes system automation events."""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        # Interpolate coordinates to full screen resolution
        x_raw = index_tip.x * self.width
        y_raw = index_tip.y * self.height
        x3 = np.clip(
            np.interp(x_raw, (self.frame_reduction, self.width - self.frame_reduction), (0, self.screen_w)),
            0,
            self.screen_w,
        )
        y3 = np.clip(
            np.interp(y_raw, (self.frame_reduction, self.height - self.frame_reduction), (0, self.screen_h)),
            0,
            self.screen_h,
        )

        fingers = self._evaluate_finger_states(landmarks, handedness)
        index_pinch = self._get_normalized_pinch_distance(
            thumb_tip, index_tip, wrist, middle_mcp, self.width, self.height
        )
        middle_pinch = self._get_normalized_pinch_distance(
            thumb_tip, middle_tip, wrist, middle_mcp, self.width, self.height
        )

        # 1. Left Click / Dragging
        if index_pinch < self.pinch_ratio_threshold:
            if not self.is_pinched:
                self.is_pinched = True
                self.pinch_start_time = current_time
                self.pinch_start_pos = (x3, y3)

            dist_moved = np.hypot(x3 - self.pinch_start_pos[0], y3 - self.pinch_start_pos[1])
            if (current_time - self.pinch_start_time > 0.25 or dist_moved > 30) and not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True

            if self.is_dragging:
                self.cloc_x = self.ploc_x + (x3 - self.ploc_x) / self.smoothing
                self.cloc_y = self.ploc_y + (y3 - self.ploc_y) / self.smoothing
                pyautogui.moveTo(self.cloc_x, self.cloc_y)
                self.ploc_x, self.ploc_y = self.cloc_x, self.cloc_y
                return "Active Drag"
            return "Pinch Hold"

        # 2. Release Pinch Event (Click Execution)
        elif self.is_pinched:
            self.is_pinched = False
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
                return "Drag Released"
            else:
                if current_time - self.last_click_time > self.click_cooldown:
                    pyautogui.click()
                    self.last_click_time = current_time
                    return "Left Click"

        # 3. Right Click
        elif middle_pinch < self.pinch_ratio_threshold:
            if current_time - self.last_click_time > self.click_cooldown:
                pyautogui.rightClick()
                self.last_click_time = current_time
                return "Right Click"

        # 4. Vertical Scroll
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            scroll_speed = int((y_raw - (self.height / 2)) / 5)
            if abs(scroll_speed) > 2:
                pyautogui.scroll(-scroll_speed)
                return f"Scrolling ({'Down' if scroll_speed > 0 else 'Up'})"

        # 5. Cursor Navigation
        elif fingers[1] == 1:
            self.cloc_x = self.ploc_x + (x3 - self.ploc_x) / self.smoothing
            self.cloc_y = self.ploc_y + (y3 - self.ploc_y) / self.smoothing
            pyautogui.moveTo(self.cloc_x, self.cloc_y)
            self.ploc_x, self.ploc_y = self.cloc_x, self.cloc_y
            return "Moving Pointer"

        return "Idle"

    def run(self):
        """Starts real-time video capture and process loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        cv2.namedWindow("Vision Control Interface", cv2.WINDOW_NORMAL)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                # Active Mousepad Region
                cv2.rectangle(
                    frame,
                    (self.frame_reduction, self.frame_reduction),
                    (self.width - self.frame_reduction, self.height - self.frame_reduction),
                    (255, 0, 255),
                    2,
                )

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                status = "Idle"
                if results.multi_hand_landmarks and results.multi_handedness:
                    landmarks = results.multi_hand_landmarks[0].landmark
                    handedness = results.multi_handedness[0].classification[0].label

                    self.mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
                    status = self._process_gestures(landmarks, handedness, time.time())

                cv2.putText(
                    frame,
                    f"State: {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Vision Control Interface", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            if self.is_dragging:
                pyautogui.mouseUp()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = GestureController()
    controller.run()