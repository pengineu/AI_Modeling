# Task2 StarGAN 학습 실패 보고 — 모드 붕괴 (RTX 4060 머신)

> 작성: GPU 실행 머신(RTX 4060). **코드 수정은 코드 담당 PC에서 진행 요망.**
> 날짜: 2026-05-30. 모델 **구조는 변경 금지**(노트북 인라인 구조·체크포인트 호환). 아래 제안은 모두 **학습 루프(train.py)·하이퍼파라미터**만 건드림.

## 1. 증상
- 본 학습(`--iters 200000 --batch 16 --img-size 128`) 도중 **iter ~14,400에서 불안정 스파이크 시작 → 회복 못 하고 iter 16,000에 모드 붕괴로 고착.**
- `samples_task2/iter_016000.png`: 1행(원본)은 정상이나, 생성 결과 3개 스타일 행(2·3·4행)이 **전부 동일한 초록 뭉개진 덩어리 + 빨간 점**으로 붕괴. 콘텐츠·스타일 구분 모두 소실.
- Task1 분류 학습은 **정상 완료**(`checkpoints/task1.pt`, val style 1.0000 / fruit 0.9993). 이 문제는 Task2 한정.

## 2. 손실 추세 (근거)
정상 구간 (iter ~5k–14.3k):
```
D ~0.2    G_adv ~2.0    G_cls <0.5    G_rec ~0.15–0.20
```
붕괴 구간 (iter 14.4k–16.7k):
```
iter 14400  D 0.019  G_adv 12.196  G_cls 14.728  G_rec 0.720
iter 14500  D 0.005  G_adv 22.086  G_cls 1.910   G_rec 0.680
iter 16500  D 0.000  G_adv 18.211  G_cls 22.098  G_rec 0.618
iter 16600  D 0.040  G_adv 33.680  G_cls 14.942  G_rec 0.551
```
- `D → 0.000` 고착 = **판별자 압승**(생성자가 받을 적대 그래디언트 소실, vanishing gradient).
- `G_adv` 폭발(최대 33), `G_rec` 0.15→0.6 = **콘텐츠 재구성 붕괴**.

## 3. 근본 원인 (진단)
판별자에 **Lipschitz 제약이 전혀 없음** + **판별자 과학습**의 조합:
- `src/task2/models.py`의 `Discriminator`: **spectral norm 없음, gradient penalty 없음** (순수 PatchGAN).
- `src/task2/train.py`: **hinge loss** 사용 + 판별자는 매 iter 업데이트, 생성자는 `n_critic=5`마다 → **D:G 업데이트 비율 5:1**.
- 원조 StarGAN도 동일한 5:1을 쓰지만 **WGAN-GP(gradient penalty)로 D를 제약**해 안정적임. 본 코드는 그 제약 없이 5:1로 D를 밀어줘서, D가 압도 → 모드 붕괴.

## 4. 추가 문제 — 체크포인트 덮어쓰기로 인한 가중치 손실
- `src/task2/train.py`가 **매 5000 iter마다 같은 파일 `checkpoints/stargan_G.pt`를 덮어씀.**
- 결과적으로 멀쩡했던 iter 10k 가중치를 잃었고, 디스크에 남은 체크포인트는 붕괴 직전(iter 15k) 상태 → **재학습 불가피.**

## 5. 제안 수정 (코드 담당 PC, 구조 변경 없이 학습 루프/하이퍼파라미터만)
우선순위 순:
1. **판별자 안정화 (필수, 택1 이상)**
   - R1 gradient penalty 추가 (StyleGAN식, hinge 붕괴의 표준 처방), **또는**
   - `Discriminator` conv에 `torch.nn.utils.spectral_norm` 적용, **또는**
   - WGAN-GP.
2. **D:G 균형**: `n_critic` 5→1, 또는 TTUR로 D의 lr만 낮춤.
3. **체크포인트 덮어쓰기 방지**: iter 태그 저장(`stargan_G_{iter:06d}.pt`) 또는 best 스냅샷 유지 — 후속 붕괴 시 좋은 가중치 보존.
4. (선택) 생성자 가중치 **EMA** → 샘플 안정화.

> 위 1~4는 모두 `train.py`(+필요시 models.py의 D에 spectral_norm 래핑)만 수정하며, **Generator의 `state_dict` 키/구조는 그대로**이므로 제출 노트북 인라인 구조·`load_state_dict`와 호환됨. (spectral_norm을 D에 적용하는 것은 G 체크포인트와 무관 — 제출은 G만 사용.)

## 6. GPU 머신 현재 상태 (재학습 준비 완료)
- 붕괴된 학습 프로세스 **중단 완료**, GPU 해제됨.
- Task1 정상: `checkpoints/task1.pt` 사용 가능.
- 데이터(`data/train` 7200장)·환경(CUDA torch 2.5.1+cu121) 준비됨.
- **코드 담당 PC에서 위 수정 후 git push → 이 머신에서 git pull 하면 동일 명령으로 즉시 재학습 가능.**

## 7. 교훈
- GAN은 **손실값이 진척도 지표가 아님** — 붕괴는 손실보다 샘플 이미지에서 먼저 명확히 드러남. 재학습 시에도 초반 샘플(iter 2k·4k…)을 일찍·자주 점검할 것.
