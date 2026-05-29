# RUNBOOK — 학습 → 업로드 → 제출

학번 202502204 · 마감 6/5(금) 23:59

> ⚠️ 이 작업 PC(노GPU)의 `torch 2.11` 빌드는 `backward()`에서 segfault 발생 → **학습 불가**. 학습은 **RTX 4060 머신**에서 안정 버전 torch(CUDA)로 진행. 이 PC에서는 코드 작성/forward 검증만 함.

---

## A. RTX 4060 머신에서 환경 준비

```bash
# 안정 버전 권장 (CUDA 빌드). 예시:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install gdown pillow
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # True 여야 함
```

이 repo를 RTX 4060 머신에 git clone/pull 후, 프로젝트 루트에서 데이터 준비:
```python
import gdown, zipfile, os
gdown.download("https://drive.google.com/uc?id=14Yc5eDlrEvx4wWp6gtSiJ52ne_noEl9u", "train.zip", quiet=False)
with zipfile.ZipFile("train.zip") as z: z.extractall("data")   # -> data/train/images, data/train/train_labels.csv
```
확인: `python -m src.common.dataset data/train`  (images=7200, 라벨 분포 균형이면 OK)

## B. 학습 (RTX 4060)

```bash
# Task 1 (분류) — 수십 분 내외
python -m src.task1.train --data data/train --epochs 15 --batch 64 --out checkpoints/task1.pt

# Task 2 (StarGAN) — 가장 오래 걸림. 먼저 짧게 동작 확인 후 본 학습.
python -m src.task2.train --data data/train --iters 2000  --batch 16 --img-size 128   # 동작/샘플 확인
python -m src.task2.train --data data/train --iters 200000 --batch 16 --img-size 128   # 본 학습
#   진행 샘플: samples_task2/iter_*.png 로 변환 품질 확인. 8GB 부족하면 --batch 8.

# Task 3 (검색): 학습 불필요. task1.pt 만 있으면 됨(라벨용). CLIP은 공개 가중치.
```
산출물: `checkpoints/task1.pt`, `checkpoints/stargan_G.pt`.

## C. Google Drive 업로드 & 링크

1. `task1.pt`, `stargan_G.pt` 를 Google Drive 업로드.
2. 각 파일 "링크가 있는 모든 사용자" 공개로 설정, 공유 링크 복사.
3. 링크에서 파일 ID 추출: `https://drive.google.com/file/d/<파일ID>/view`.
4. `Termproject_202502204_Task{1,2,3}_model.txt` 에 링크/ID 기입.

## D. 제출 노트북 설정 (Colab)

각 노트북의 **설정 셀**에 파일 ID 입력:
- Task1: `MODEL_FILE_ID = task1.pt ID`
- Task2: `STARGAN_FILE_ID = stargan_G.pt ID`, `TASK1_FILE_ID = task1.pt ID`
- Task3: `MODEL_FILE_ID = task1.pt ID`

test 데이터: TA가 `data/test/images/` 에 두거나, test.zip의 gdrive ID를 `TEST_ZIP_ID` 에 지정하면 자동 처리.
(노트북은 `data/test` 가 없을 때만 다운로드하며, 이미지 탐색은 재귀적이라 구조가 조금 달라도 동작.)

## E. 제출 전 검증 (무료 Colab 새 런타임)

각 노트북을 위→아래 전체 실행하여 오류 없이 아래 생성 확인:
- `release/202502204.test.task1.txt` — 줄 수 = test 이미지 수, 탭 구분, style∈0–2 / fruit∈0–5
- `release/202502204.test.task2/` — 이미지당 2개, `{num}_generate_{style}.jpg`, 원본 style 미포함
- `release/202502204.test.task3.txt` — query당 `[(id,style,fruit),...]`, `TOP_K` 변경 반영

## F. 최종 제출물 (사이버캠퍼스)

- `Termproject_202502204_Task1.ipynb`, `_Task2.ipynb`, `_Task3.ipynb`
- `Termproject_202502204_Task1_model.txt`, `_Task2_model.txt`, `_Task3_model.txt`
- `202502204.zip`  ← 구성 모호. TA(goun.pyeon@o.cnu.ac.kr) 확인 권장. (잠정: 위 노트북 3 + model.txt 3, 필요시 release 출력 포함)
