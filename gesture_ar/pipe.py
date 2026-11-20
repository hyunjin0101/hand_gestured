"""Two-hand pipe overlay with movement-triggered horizontal flip."""
import time
import cv2
import numpy as np
from .common import (require_image,make_sound,play,overlay,palm_center,
                     frame_points,open_camera,read_frame,process_hands,show_label)


def pipe_anchors(hands, frame_shape):
    thumbs = [frame_points(h,[4],frame_shape)[0] for h in hands]
    pinkies = [frame_points(h,[20],frame_shape)[0] for h in hands]
    return min(thumbs,key=lambda pt:pt[1]),max(pinkies,key=lambda pt:pt[1])


def render_pipe(frame, top, bottom, sprite, flipped=False):
    distance = np.linalg.norm(bottom-top)
    if distance < 5:return
    width = max(1,int(sprite.shape[1]*(distance/sprite.shape[0])))
    height = max(1,int(distance))
    # Guard against very large sprites when hands are far apart.
    width = min(width,frame.shape[1]*3)
    height = min(height,frame.shape[0]*3)
    resized = cv2.resize(sprite,(width,height))
    if flipped:resized = cv2.flip(resized,1)
    midpoint = (top+bottom)/2
    overlay(frame,resized,int(midpoint[0]-width/2),int(midpoint[1]-height/2),0.5)


def run(camera=0, movement_threshold=30.0):
    import mediapipe as mp
    sprite = require_image('pipe.png')
    sound = make_sound('pipe.mp3')
    mp_hands = mp.solutions.hands
    cap = open_camera(camera)
    prev_center=None;flipped=False;last_flip=0.0;was_visible=False
    try:
        with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            while True:
                frame = read_frame(cap)
                if frame is None: break
                result=process_hands(hands,frame)
                objects=result.multi_hand_landmarks or []
                visible=len(objects)>=2
                if visible:
                    landmarks=[o.landmark for o in objects[:2]]
                    current_center=palm_center(landmarks[0],frame.shape)
                    now=time.monotonic()
                    if (prev_center is not None
                        and np.linalg.norm(current_center-prev_center)>movement_threshold
                        and now-last_flip>0.35):
                        flipped=not flipped;last_flip=now
                    prev_center=current_center
                    if not was_visible:play(sound)
                    top,bottom=pipe_anchors(landmarks,frame.shape)
                    render_pipe(frame,top,bottom,sprite,flipped)
                    for o in objects:mp.solutions.drawing_utils.draw_landmarks(frame,o,mp_hands.HAND_CONNECTIONS)
                else:prev_center=None
                was_visible=visible
                show_label(frame,f'PIPE flipped: {flipped} | Q: quit')
                cv2.imshow('Pipe AR',frame)
                if cv2.waitKey(1)&0xFF==ord('q'):break
    finally:
        cap.release();cv2.destroyAllWindows()
