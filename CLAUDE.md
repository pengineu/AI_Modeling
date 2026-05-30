# CLAUDE.md — RTX 4060 데스크톱 작업 지침

이 파일은 **RTX 4060(8GB) 데스크톱의 Claude Code**를 위한 지침이다. 이 머신의 임무는 **모델 학습**과 **검증**이다. 코드 작성·노트북은 다른 PC에서 이미 완료되어 git으로 동기화되었다.

## 프로젝트 한눈에
- 과목 텀프로젝트: 과일 이미지(6 class × 3 style × 400 = 7200장) 3개 태스크.
- 학번 `202502204`, 마감 **6/5(금) 23:59**.
- Task1 분류(50) · Task2 StarGAN 스타일변환(50) · Task3 CLIP 검색(20).
- 제출 노트북은 **무료 Colab 추론 전용**. 학습된 모델은 Google Drive 링크로 제공.
- **이 머신의 역할**: `src/task1/train.py`, `src/task2/train.py` 를 GPU로 돌려 `checkpoints/task1.pt`, `checkpoints/stargan_G.pt` 생성 → 업로드 → 노트북에 ID 기입.

## ⚠️ 환경 (가장 먼저 확인)
- **다른 작업 PC의 torch 2.11(Windows CPU)은 `backward()`에서 segfault** 가 있었다. 그건 그 머신만의 결함이고, **이 RTX 머신에서는 안정 버전 CUDA torch를 설치**하면 정상이다.
- 권장 설치:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install gdown pillow
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True 필수
  ```
- 학습 시작 전 **백워드 동작 확인**: `python -c "import torch,torch.nn as nn; nn.Linear(4,2)(torch.randn(2,4)).sum().backward(); print('backward OK')"`

## 데이터 준비
```python
import gdown, zipfile
gdown.download("https://drive.google.com/uc?id=14Yc5eDlrEvx4wWp6gtSiJ52ne_noEl9u", "train.zip", quiet=False)
with zipfile.ZipFile("train.zip") as z: z.extractall("data")
```
**확정 구조**(추정 금지): `data/train/images/{0..7199}.jpg` (플랫) + `data/train/train_labels.csv`(헤더 `file_name,style,fruit`).
확인: `python -m src.common.dataset data/train` → images=7200, fruit 6×1200 / style 3×2400 균형이면 정상.
- Fruit: 0 apple·1 asian pear·2 banana·3 cherry·4 grape·5 pineapple
- Style: 0 pencil color·1 oil painting·2 water color

## 학습 명령
```bash
# Task 1 (분류) — 수십 분
python -m src.task1.train --data data/train --epochs 15 --batch 64 --out checkpoints/task1.pt
#   목표: val style acc 거의 1.0, fruit acc 최대화. 과적합 시 epoch/aug 조정.

# Task 2 (StarGAN) — 가장 오래 걸림. 먼저 짧게 동작 확인.
python -m src.task2.train --data data/train --iters 2000   --batch 16 --img-size 128   # 동작·샘플 확인
python -m src.task2.train --data data/train --iters 200000 --batch 16 --img-size 128   # 본 학습
#   OOM(8GB 초과) 시 --batch 8. 진행 품질은 samples_task2/iter_*.png 로 눈으로 확인.

# Task 3: 학습 불필요(CLIP 공개 가중치 + task1.pt 재사용).
```

## 절대 바꾸면 안 되는 것 (제출 호환성)
1. **라벨 정의·출력 형식** — `src/common/labels.py` 및 노트북 출력 형식(Task1 `img_id\tstyle\tfruit`, Task2 `{num}_generate_{style}.jpg`, Task3 `[(id,style,fruit),...]`).
2. **모델 아키텍처 일치** — 제출 노트북 3개(`notebooks/*.ipynb`)에는 모델 구조가 **인라인 복제**되어 있다. `src/task1/model.py`(DualHeadClassifier)나 `src/task2/models.py`(Generator)의 구조(채널·레이어·`img_size`·정규화)를 바꾸면 **노트북 인라인 코드와 체크포인트도 같이 수정**해야 한다. 안 그러면 `load_state_dict` 실패.
   - Task1 전처리: Resize(img_size=224), ImageNet 정규화.
   - Task2 전처리: Resize(128), Normalize([0.5]*3,[0.5]*3) → Tanh 출력.
3. 체크포인트 저장 형식: `torch.save({"model": state_dict, "img_size": ..., ...})`. 노트북이 `ckpt["model"]`을 읽는다.

## 학습 후 (업로드 & 기입)
1. `checkpoints/task1.pt`, `checkpoints/stargan_G.pt` → Google Drive 업로드, **"링크 있는 모든 사용자" 공개**.
2. 링크의 파일 ID(`/file/d/<ID>/view`)를 추출.
3. 루트의 `Termproject_202502204_Task{1,2,3}_model.txt` 에 링크·ID 기입.
4. 노트북 설정 셀의 `*_FILE_ID` 채우기: Task1=`MODEL_FILE_ID`, Task2=`STARGAN_FILE_ID`+`TASK1_FILE_ID`, Task3=`MODEL_FILE_ID`.

## 검증 (가능하면 무료 Colab 새 런타임)
각 노트북 위→아래 전체 실행, 오류 없이 생성 확인:
- `release/202502204.test.task1.txt` (줄 수=이미지 수, 탭 구분, style∈0–2/fruit∈0–5)
- `release/202502204.test.task2/` (이미지당 2개, 원본 style 제외)
- `release/202502204.test.task3.txt` (query당 Top-K 튜플)

## 파일 맵
```
src/common/{labels.py, dataset.py}     # 라벨 고정 / CSV 파싱 (검증 완료)
src/task1/{model.py, train.py}         # 분류
src/task2/{models.py, train.py}        # StarGAN (hinge loss)
notebooks/Termproject_202502204_Task{1,2,3}.ipynb   # 제출 추론(자기완결, 모델 인라인)
checkpoints/                           # 학습 산출물(gitignore)
RUNBOOK.md                             # 사람용 전체 절차
```

## 진행 상태
- [x] 데이터 구조 확정, common 모듈, 모델·학습 스크립트, 노트북 3개, model.txt 템플릿 (작성·forward 검증 완료)
- [ ] **이 머신**: Task1 학습 → Task2 학습 → 체크포인트 업로드 → ID 기입 → Colab 검증
