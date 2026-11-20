"""Shared camera, asset, audio, and RGBA compositing utilities."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "assets" / "images"
SOUNDS = ROOT / "assets" / "sounds"


def require_image(name: str):
    path = IMAGES / name
    if not path.is_file():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {path}\nREADME의 assets 목록을 확인하세요.")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def make_sound(name: str):
    """Audio is optional. Failed initialization must not disable vision."""
    path = SOUNDS / name
    if not path.is_file():
        print(f"[INFO] 선택적 사운드가 없어 무음으로 실행합니다: {path.name}")
        return None
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return pygame.mixer.Sound(str(path))
    except (ImportError, OSError, Exception) as exc:
        print(f"[WARN] 사운드 재생 비활성화 ({path.name}): {exc}")
        return None


def play(sound):
    if sound is not None:
        sound.play()


def overlay(frame: np.ndarray, sprite: np.ndarray, x: int, y: int, alpha_multiplier: float = 1.0):
    """Alpha blend BGR/BGRA sprite at (x,y), safely clipping at all frame edges."""
    if sprite is None:
        return
    h, w = sprite.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(frame.shape[1], int(x)+w), min(frame.shape[0], int(y)+h)
    if x1 >= x2 or y1 >= y2:
        return
    rx1, ry1 = x1-int(x), y1-int(y)
    cut = sprite[ry1:ry1+(y2-y1), rx1:rx1+(x2-x1)]
    if cut.shape[2] == 4:
        alpha = (cut[:, :, 3:4].astype(np.float32)/255) * alpha_multiplier
    else:
        alpha = np.full((y2-y1,x2-x1,1), alpha_multiplier, np.float32)
    front = cut[:, :, :3].astype(np.float32)
    back = frame[y1:y2,x1:x2].astype(np.float32)
    frame[y1:y2,x1:x2] = np.clip(front*alpha + back*(1-alpha),0,255).astype(np.uint8)


def palm_center(hand, frame_shape):
    idx = (0,5,9,13,17)
    w,h = frame_shape[1], frame_shape[0]
    return np.array([np.mean([hand[i].x for i in idx])*w,
                     np.mean([hand[i].y for i in idx])*h])


def frame_points(hand, indices, shape):
    return [np.array((hand[i].x*shape[1],hand[i].y*shape[0])) for i in indices]


def open_camera(index: int):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        cam.release()
        raise RuntimeError(f"웹캠 {index}을 열 수 없습니다. 카메라 권한과 인덱스를 확인하세요.")
    return cam


def show_label(frame, msg: str):
    cv2.putText(frame,msg,(12,32),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2,cv2.LINE_AA)


def read_frame(cam):
    ok, frame = cam.read()
    if not ok:
        return None
    return cv2.flip(frame,1)


def process_hands(hands, frame):
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return hands.process(rgb)
