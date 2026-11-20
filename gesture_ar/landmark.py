"""Pinch-triggered effect sprites; cooldown 1s, lifetime 2s."""
import time
import cv2
import numpy as np
from .common import (require_image,make_sound,play,overlay,
                     open_camera,read_frame,process_hands,show_label)


def render_effects(frame, active, sprites, now):
    """Draw current active effects and return only those not yet expired."""
    remaining=[]
    for idx,pos,until in active:
        if now>=until:continue
        sprite=sprites[idx]
        overlay(frame,sprite,int(pos[0]-sprite.shape[1]//2),int(pos[1]-sprite.shape[0]))
        remaining.append((idx,pos,until))
    return remaining


def run(camera=0, cooldown=1.0, duration=2.0):
    import mediapipe as mp
    sprites=[cv2.resize(require_image(f'landmark_{i}.png'),(150,150)) for i in range(5)]
    sound=make_sound('landmark.mp3')
    mp_hands=mp.solutions.hands
    cap=open_camera(camera)
    active=[];current_index=0;last_activation=float('-inf')
    try:
        with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.7,
                            min_tracking_confidence=0.5) as hands:
            while True:
                frame=read_frame(cap)
                if frame is None:break
                now=time.monotonic()
                result=process_hands(hands,frame)
                if result.multi_hand_landmarks:
                    for item in result.multi_hand_landmarks:
                        lm=item.landmark
                        # Same normalized-thumb/index distance threshold as original.
                        dist=float(np.hypot(lm[8].x-lm[4].x,lm[8].y-lm[4].y))
                        if dist<0.05 and now-last_activation>cooldown:
                            pos=(int(lm[8].x*frame.shape[1]),int(lm[8].y*frame.shape[0]))
                            active.append((current_index,pos,now+duration))
                            current_index=(current_index+1)%len(sprites)
                            last_activation=now
                            play(sound)
                        mp.solutions.drawing_utils.draw_landmarks(frame,item,mp_hands.HAND_CONNECTIONS)
                active=render_effects(frame,active,sprites,now)
                show_label(frame,'LANDMARK pinch | Q: quit')
                cv2.imshow('Landmark AR',frame)
                if cv2.waitKey(1)&0xFF==ord('q'):break
    finally:
        cap.release();cv2.destroyAllWindows()
