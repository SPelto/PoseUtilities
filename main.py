import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller

keyboard = Controller()


def changeVolume(direction):
    if direction == "up":
        keyboard.press(Key.media_volume_up)
        keyboard.release(Key.media_volume_up)
    if direction == "down":
        keyboard.press(Key.media_volume_down)
        keyboard.release(Key.media_volume_down)
    return


def index_thumb_dist(hand_landmarks):
    index_finger_coord = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
    ]

    thumb_coord = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
    ]

    dist = np.linalg.norm(np.array(index_finger_coord) - np.array(thumb_coord))
    return dist


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_index_dist = index_thumb_dist(hand_landmarks)

                if thumb_index_dist < 0.05:
                    print("Raise Volume", f"Thumb & Index distance: {thumb_index_dist}")
                    changeVolume("up")
                elif thumb_index_dist > 0.25:
                    print(
                        "Lower volume",
                        f"Thumb & Index distance: {thumb_index_dist}",
                    )
                    changeVolume("down")

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
        if cv2.waitKey(50) & 0xFF == 27:
            break
cap.release()
