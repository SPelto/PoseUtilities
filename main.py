import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import gesture_utils, system_controls

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

# fig = plt.figure()
application = None
action = None
t = -1000
first_occurence = None
gesture_time_check = None
time_threshold = 1
with mp_hands.Hands(
    model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Do gesture recognition
                application_candidate, action = gesture_utils.choose_action(
                    hand_landmarks
                )

                # Check that the same action has been on for at least time_threshold seconds
                if gesture_time_check is None:
                    gesture_time_check = application_candidate
                    first_occurence = time.time()
                    continue

                if application_candidate != gesture_time_check:
                    gesture_time_check = application_candidate
                    first_occurence = time.time()
                    continue

                if (
                    application_candidate == gesture_time_check
                    and time.time() - first_occurence < time_threshold
                ):
                    continue

                # Make sure that commandline calls are not executed more often than once in 30 seconds
                if application_candidate == "Commandline call":
                    if time.time() - t > 30:
                        t = time.time()
                        application = application_candidate
                    else:
                        application = None
                else:
                    application = application_candidate

                print(application, action)
                system_controls.application(application, action)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))

        if cv2.waitKey(50) & 0xFF == 27:
            break

cap.release()
