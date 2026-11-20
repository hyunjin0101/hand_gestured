"""Tiger paw on a palm; size follows distance from palm center to index tip."""
import cv2
import numpy as np
from .common import (require_image, make_sound, play, overlay, palm_center,
                     open_camera, read_frame, process_hands, show_label)


def render_tiger(frame, hand, sprite):
    center = palm_center(hand,frame.shape)
    tip = np.array((hand[8].x*frame.shape[1], hand[8].y*frame.shape[0]))
    size = int(np.clip(np.linalg.norm(tip-center)*1.8, 80,180))
    paw = cv2.resize(sprite,(size,size))
    overlay(frame,paw,int(center[0]-size/2),int(center[1]-size/2))


def run(camera=0):
    import mediapipe as mp
    sprite = require_image('tigerpaw.png')
    sound = make_sound('tiger.mp3')
    mp_hands = mp.solutions.hands
    cap = open_camera(camera)
    was_visible = False
    try:
        with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            while True:
                frame = read_frame(cap)
                if frame is None: break
                result = process_hands(hands,frame)
                visible = bool(result.multi_hand_landmarks)
                if visible:
                    if not was_visible: play(sound)
                    for item in result.multi_hand_landmarks:
                        render_tiger(frame,item.landmark,sprite)
                        mp.solutions.drawing_utils.draw_landmarks(frame,item,mp_hands.HAND_CONNECTIONS)
                was_visible = visible
                show_label(frame,'TIGER | Q: quit')
                cv2.imshow('Tiger Paw AR',frame)
                if cv2.waitKey(1)&0xFF==ord('q'):break
    finally:
        cap.release();cv2.destroyAllWindows()
