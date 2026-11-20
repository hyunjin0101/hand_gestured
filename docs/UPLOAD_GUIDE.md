# GitHub 업로드 가이드 (Windows)

## 설치 및 실행
1. 저장소를 clone하거나 ZIP으로 다운로드한 뒤 프로젝트 루트에서 실행합니다.
2. Python 3.10을 준비하고 README의 명령으로 가상환경과 의존성을 설치합니다.
3. 이미지·사운드 리소스는 소유권과 공개 권한을 확인한 후 assets/에 추가합니다.
4. 모델 추론 모드에는 학습된 .h5 파일과 추가 TensorFlow 의존성이 필요합니다.

## Git으로 수정사항 업로드

```powershell
git clone https://github.com/hyunjin0101/hand_gestured.git
cd hand_gestured
# 수정 후
git add .
git commit -m "Describe changes"
git push origin main
```

## 업로드 전 확인
- 공개하면 안 되는 데이터, 개인정보, 토큰, 모델 가중치가 없는지 점검합니다.
- .gitignore에서 제외된 파일도 이미 추적 중이라면 별도로 제거해야 합니다.
- 팀 프로젝트의 본인 담당 부분을 README에 정확히 기술합니다.

## 프로젝트 README와 프로필 README
프로젝트 설명은 이 저장소 README에 작성합니다. GitHub 프로필 전용 README는 GitHub 사용자명과 동일한 이름의 공개 저장소에 만듭니다.
