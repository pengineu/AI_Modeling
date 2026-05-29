# AI_Modeling — Term Project (Classification & Generation)

학번 `202502204` · 마감 6/5(금) 23:59

과일 이미지(6 class × 3 style × 400 = 7200장)에 대한 3개 태스크.

| Task | 내용 | 배점 | 출력물 |
|------|------|------|--------|
| 1 | 분류 (과일 6 + 스타일 3) | 50 | `release/202502204.test.task1.txt` |
| 2 | StarGAN 스타일 변환 | 50 | `release/202502204.test.task2/` |
| 3 | CLIP 임베딩 Top-K 검색 | 20 | `release/202502204.test.task3.txt` |

## 라벨 (고정)
- Fruit: 0 apple · 1 asian pear · 2 banana · 3 cherry · 4 grape · 5 pineapple
- Style: 0 pencil color · 1 oil painting · 2 water color

## 작업 흐름
1. **학습 (RTX 4060 데스크탑)**: `src/task*/train.py`, `build_gallery.py` 실행 → `checkpoints/`.
2. **모델 업로드**: `checkpoints/*` → Google Drive → 공유 링크를 `Termproject_202502204_Task{N}_model.txt`에 기재.
3. **제출 (무료 Colab)**: `notebooks/Termproject_202502204_Task{N}.ipynb` 실행 → gdrive에서 데이터/모델 받아 추론 → `release/` 출력.

> 제출 노트북은 **추론(test) 전용**. 학습 코드는 제출 불필요. 모든 `pip install`은 `==`로 버전 고정.

## 디렉터리
```
src/common/   labels.py(라벨 고정), dataset.py(폴더→라벨 파싱)
src/task1/    model.py(EfficientNet-B0 2-head), train.py
src/task2/    models.py(StarGAN G/D), train.py
src/task3/    build_gallery.py(CLIP 갤러리 사전계산)
notebooks/    제출용 Colab 추론 노트북 3개
checkpoints/  학습 산출물(gitignore) → gdrive 업로드
release/      출력물(gitignore)
data/         데이터셋(gitignore)
```

## 데이터 준비
```python
import gdown, zipfile, os
gdown.download("https://drive.google.com/uc?id=14Yc5eDlrEvx4wWp6gtSiJ52ne_noEl9u", "train.zip", quiet=False)
with zipfile.ZipFile("train.zip") as z: z.extractall("data")
```
평가 시에는 `data/test`(구조 동일, TA 제공)로 대체.
