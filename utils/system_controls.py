from pynput.keyboard import Key, Controller
from utils import gesture_utils
import subprocess
import os
import time

keyboard = Controller()


def changeVolume(direction):
    if direction == "up":
        keyboard.press(Key.media_volume_up)
        keyboard.release(Key.media_volume_up)

    if direction == "down":
        keyboard.press(Key.media_volume_down)
        keyboard.release(Key.media_volume_down)
    return


def application(application, action):
    if application == "volume":
        changeVolume(action)
    if application == "Commandline call":
        subprocess.Popen(["python", action])
    return
