# Gesture-Driven AR Effects 🎥✋

**MediaPipe 기반 실시간 손 추적·제스처 분류 및 AR 이펙트 시스템**  
*Real-Time Hand Tracking, Gesture Recognition & Interactive AR Effects*

> **포트폴리오 소개**  
> 정보통신공학을 전공하며 컴퓨터 비전과 실시간 인터랙티브 시스템에 관심을 갖고 있습니다. 이 프로젝트에서는 손 랜드마크를 이용해 손동작을 분석하고, 화면 좌표에 맞춰 이미지·사운드 효과를 합성하는 기능을 구현·개선했습니다. 특히 제스처별 랜드마크 활용 방식과 판단 임계값을 조정하고, 개별 효과 모듈의 출력 및 동작 오류를 확인·수정하는 과정에 참여했습니다.

## 프로젝트 개요

- **프로젝트 형태:** 팀 프로젝트(4인 참여, 사용자 설명 기준)
- **진행 시기:** 2025년 6월 (기존 프로젝트 이력 기준)
- **목표:** 웹캠 영상 속 손동작에 반응하는 실시간 AR 시각·사운드 효과
- **기술:** Python, OpenCV, MediaPipe Hands, NumPy, SciPy, Pygame, TensorFlow/Keras(선택)
- **담당:** 효과별 코드 수정·디버깅, 특징점 활용·판단 임계값 개선, 효과 모듈 연동 및 출력 검증

### 주요 기능

| 효과 | 동작 방식 | 구현 포인트 |
|---|---|---|
| **Tiger Paw** | 손바닥 중앙에 발 이미지를 합성 | 손바닥–검지 끝 거리 기반 크기 조절, 알파 블렌딩 |
| **Pipe** | 두 손의 엄지·새끼손가락 좌표를 기준으로 이미지 배치 | 양손 위치를 이용한 크기 조절, 움직임 기반 좌우 반전 |
| **Landmark** | 엄지·검지 끝이 가까워질 때 효과 발동 | 5개 이미지 순환, 1초 쿨다운, 2초 지속, 사운드 |
| **Model** *(선택)* | 학습된 분류 모델에서 `pipe`, `tiger`, `landmarks` 예측 | 프레임당 198 특징, 30프레임 입력, 임계값·3회 일치 확인 |

> **중요:** `model` 모드는 사용자 제공 학습 모델(`.h5`)이 추가되어야 합니다. `tiger`, `pipe`, `landmark` 모드는 해당 이미지 리소스가 있으면 모델 없이 개별 시연이 가능합니다. 이 저장소에는 모델 가중치와 미디어 리소스를 포함하지 않았습니다.

## 처리 흐름

```text
Webcam frame
    │
    ▼
MediaPipe Hands (21 landmarks per hand, up to 2 hands)
    │
    ├── Direct effect modes
    │     ├── Tiger: palm center + hand openness
    │     ├── Pipe: two-hand anchors + movement threshold
    │     └── Landmark: index/thumb distance + cooldown
    │
    └── Optional model mode
          └── 84 landmark values + 15 angles + 99 distances
                    → 30-frame sequence → Keras 3-class inference
    │
    ▼
OpenCV image compositing + optional Pygame audio
    │
    ▼
Live preview
```

## 디렉터리 구조

```text
gesture-ar-effects/
├── main.py                       # 실행 진입점
├── gesture_ar/
│   ├── common.py                 # 웹캠·이미지·사운드·합성 공통 함수
│   ├── features.py               # 모델용 198차원 특징 추출
│   ├── tiger.py                   # 호랑이 효과
│   ├── pipe.py                    # 양손 파이프 효과
│   ├── landmark.py                # 손가락 이펙트
│   └── classifier.py              # .h5 모델 추론(선택)
├── assets/
│   ├── images/                   # 이미지 직접 추가
│   └── sounds/                   # MP3 직접 추가 (선택)
├── models/                       # 학습 모델 직접 추가 (선택)
├── notebooks/effect_experiments.ipynb # 원본 Notebook(출력 로그 제거)
├── tests/test_features.py        # 특징 벡터·알파 합성 테스트
├── requirements.txt
├── requirements-model.txt
└── .gitignore
```

## 설치 및 실행 (Windows PowerShell)

Python **3.10 (64-bit)** 환경을 권장합니다. 프로젝트 루트 폴더에서 실행합니다.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

별도 이미지 파일이 필요합니다. **원본 Notebook에는 이미지·사운드·학습 모델 파일이 포함되어 있지 않습니다.** 본인이 제작했거나 공개 권한이 있는 리소스만 추가하세요.

```text
assets/images/tigerpaw.png
assets/images/pipe.png
assets/images/landmark_0.png
assets/images/landmark_1.png
assets/images/landmark_2.png
assets/images/landmark_3.png
assets/images/landmark_4.png

assets/sounds/tiger.mp3         (선택)
assets/sounds/pipe.mp3          (선택)
assets/sounds/landmark.mp3      (선택)
models/model1.h5               (모델 모드에서만 필요)
```

실행 명령:

```powershell
python main.py --mode tiger
python main.py --mode pipe
python main.py --mode landmark
```

학습된 모델을 확보한 경우에만:

```powershell
python -m pip install -r requirements-model.txt
python main.py --mode model --model-path models/model1.h5 --confidence 0.8
```

카메라가 다른 장치에 연결된 경우 `--camera 1`처럼 변경할 수 있습니다. 종료는 **Q** 키입니다.

### 테스트

```powershell
python -m unittest discover -s tests -v
```

## 구현 과정에서 확인한 문제와 개선

1. **인식 결과가 특정 동작에 치우치는 문제:** 기존 프로젝트에서 임계값과 사용 랜드마크를 수정해 비교했습니다. 개선 전후 성능 수치는 추가 측정이 필요합니다.
2. **여러 실험이 단일 Notebook에 혼재:** 효과별 독립 모듈과 공통 합성 함수로 분리했습니다.
3. **모델 출력의 순간적인 변화:** 모델 모드에서 신뢰도 임계값(기본 `0.8`)과 최근 예측 3회 일치 조건을 활용합니다.
4. **화면 경계와 이미지 투명도:** 공통 알파 합성 함수에 이미지 클리핑을 적용했습니다.
5. **반복되는 사운드 출력:** 손 감지 전환·이벤트 중심으로 사운드 재생을 제한했습니다.

## 검증 범위 및 한계

- 노트북에서 확인 가능한 기능을 기반으로 코드 구조를 정리했습니다.
- 단위 테스트는 특징 벡터 형태 및 합성 경계 처리만 검증합니다.
- 웹캠과 실제 미디어 리소스, 모델 파일을 사용한 최종 실기기 구동은 **아직 검증되지 않았습니다**.
- 198차원 특징 구성과 3개 클래스 순서는 Notebook 기준입니다. 실제 `.h5` 모델이 같은 전처리·클래스 순서로 학습되었는지 확인해야 합니다.
- 최신 MediaPipe에서는 `mp.solutions`가 제거되어 기존 예제와 호환되지 않을 수 있으므로, 이 프로젝트는 `mediapipe==0.10.21`을 사용합니다.

## 포트폴리오 개선 예정

- [ ] 실제 실행 화면 GIF 또는 짧은 시연 영상 추가 (`docs/`)
- [ ] 클래스별 정확도·오분류 사례 및 개선 전후 비교 표 추가
- [ ] 모델 가중치의 입력·출력 구조와 학습 환경 확인
- [ ] 팀 전체 구현과 본인 담당 범위를 명확히 기술

---

**Contact / Portfolio**: 본인 GitHub 프로필 또는 이메일 링크를 여기에 추가하세요.  
**Project note:** 원본 실험은 `notebooks/effect_experiments.ipynb`에서 확인할 수 있습니다. 업로드 전 모든 이미지·오디오·모델의 공개 권한을 확인하세요.