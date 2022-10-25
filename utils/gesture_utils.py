import cv2
import mediapipe as mp
import numpy as np
import os
import time
from dotenv import load_dotenv, dotenv_values

mp_hands = mp.solutions.hands


def findMaxDist(coordinates):
    maxDist = 0
    for xx in coordinates:
        for yy in coordinates:
            dist = np.linalg.norm(np.array(xx) - np.array(yy))
            if dist > maxDist:
                maxDist = dist
    # print(maxDist)
    return maxDist


def hand_size(hand_landmarks):
    indexF = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
    ]
    pinkyF = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
    ]
    wrist = [
        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
    ]
    coordinates = [indexF, pinkyF, wrist]

    return findMaxDist(coordinates)


def hand_is_open(hand_landmarks, handSize):
    middle_tip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
    ]
    ring_tip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
    ]
    pinky_tip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
    ]

    middle_knuckle = [
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
    ]
    ring_knuckle = [
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
    ]
    pinky_knuckle = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
    ]

    tips = [middle_tip, ring_tip, pinky_tip]
    knuckles = [middle_knuckle, ring_knuckle, pinky_knuckle]

    minDist = 1000

    for tip in tips:
        for knuckle in knuckles:
            dist = np.linalg.norm(np.array(tip) - np.array(knuckle))

            if dist < 0.2 * handSize:
                return False

    return True


def index_thumb_dist(hand_landmarks, handSize):
    index_finger_coord = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
    ]

    thumb_coord = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
    ]

    dist = np.linalg.norm(np.array(index_finger_coord) - np.array(thumb_coord))
    return dist / hand_size(hand_landmarks)


def V_sign(hand_landmarks, handSize):
    index_tip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
    ]
    middle_tip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
    ]
    ring_tip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
    ]

    pinky_tip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
    ]

    ring_knuckle = [
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
    ]
    pinky_knuckle = [
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
    ]

    IT_MT_dist = np.linalg.norm(np.array(index_tip) - np.array(middle_tip)) / handSize
    PT_PK_dist = (
        np.linalg.norm(np.array(pinky_tip) - np.array(pinky_knuckle)) / handSize
    )
    RT_RK_dist = np.linalg.norm(np.array(ring_tip) - np.array(ring_knuckle)) / handSize
    MT_RT_dist = np.linalg.norm(np.array(middle_tip) - np.array(ring_tip)) / handSize

    if IT_MT_dist < 0.4:
        print(f"{IT_MT_dist=}")
        return False
    if PT_PK_dist > 0.3:
        print(f"{PT_PK_dist=}")
        return False
    if RT_RK_dist > 0.3:
        print(f"{RT_RK_dist=}")
        return False
    if MT_RT_dist < IT_MT_dist:
        print(f"{MT_RT_dist=}")
        return False
    return True


def choose_action(hand_landmarks):
    load_dotenv()
    handSize = hand_size(hand_landmarks)
    thumb_index_dist = index_thumb_dist(hand_landmarks, handSize)
    hand_isOpen = hand_is_open(hand_landmarks, handSize)

    action = None
    application = None
    if thumb_index_dist < 0.285 and hand_isOpen:
        application = "volume"
        action = "up"
    elif thumb_index_dist > 1.0 and hand_isOpen:
        application = "volume"
        action = "down"
    elif V_sign(hand_landmarks, handSize):
        application = "Commandline call"
        action = os.environ.get("GESTURE")
    return application, action
