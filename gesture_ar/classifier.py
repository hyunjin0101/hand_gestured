"""Optional 3-class Keras inference; requires original compatible .h5 file."""
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from .common import (ROOT, require_image, make_sound, play, overlay, open_camera,
                     read_frame, process_hands, show_label)
from .features import extract_features
from .tiger import render_tiger

ACTIONS=('pipe','tiger','landmarks')


def run(camera=0, model_path=None, min_confidence=0.8):
    import mediapipe as mp
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError('모델 추론 모드에는 requirements-model.txt 설치가 필요합니다.') from exc

    model_file=Path(model_path) if model_path else ROOT/'models'/'model1.h5'
    if not model_file.is_file():
        raise FileNotFoundError(f'모델 파일이 없습니다: {model_file}\nREADME의 모델 파일 안내를 확인하세요.')
    model=load_model(str(model_file),compile=False)
    # Known training specification from the notebook.
    input_shape=model.input_shape
    output_shape=model.output_shape
    if (len(input_shape)!=3 or input_shape[-2:]!=(30,198)
        or output_shape[-1]!=3):
        raise ValueError(f'모델 입력/출력이 예상 형식과 다릅니다. input={input_shape}, output={output_shape}; 필요=(None,30,198)->(None,3)')

    sprites={
        'pipe':require_image('pipe.png'),
        'tiger':require_image('tigerpaw.png'),
        'landmarks':require_image('landmark_0.png')
    }
    sounds={key:make_sound(name) for key,name in
            (('pipe','pipe.mp3'),('tiger','tiger.mp3'),('landmarks','landmark.mp3'))}
    cap=open_camera(camera)
    history=deque(maxlen=30);stable=deque(maxlen=3);prev_action=None
    mp_hands=mp.solutions.hands
    try:
        with mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            while True:
                frame=read_frame(cap)
                if frame is None:break
                result=process_hands(hands,frame)
                if result.multi_hand_landmarks:
                    # One hand per temporal sequence: do not interleave two hands.
                    item=result.multi_hand_landmarks[0]
                    history.append(extract_features(item))
                    if len(history)==30:
                        x=np.expand_dims(np.array(history,dtype=np.float32),axis=0)
                        pred=np.asarray(model.predict(x,verbose=0)).reshape(-1)
                        idx=int(np.argmax(pred));confidence=float(pred[idx])
                        action=ACTIONS[idx] if confidence>=min_confidence else None
                        stable.append(action)
                        chosen=action if len(stable)==3 and all(a==action for a in stable) else None
                        if chosen:
                            if chosen!=prev_action:play(sounds[chosen])
                            if chosen=='tiger':render_tiger(frame,item.landmark,sprites['tiger'])
                            else:
                                src=sprites[chosen]
                                resized=cv2.resize(src,(100,100))
                                wrist=item.landmark[0]
                                x0=int(wrist.x*frame.shape[1]-50)
                                y0=int(wrist.y*frame.shape[0]-50)
                                overlay(frame,resized,x0,y0,0.7)
                            show_label(frame,f'{chosen.upper()} {confidence:.2f} | Q: quit')
                        else:show_label(frame,f'CLASSIFY {confidence:.2f} | Q: quit')
                        prev_action=chosen
                    mp.solutions.drawing_utils.draw_landmarks(frame,item,mp_hands.HAND_CONNECTIONS)
                else:
                    history.clear();stable.clear();prev_action=None
                    show_label(frame,'NO HAND | Q: quit')
                cv2.imshow('Gesture Classifier + AR',frame)
                if cv2.waitKey(1)&0xFF==ord('q'):break
    finally:
        cap.release();cv2.destroyAllWindows()
