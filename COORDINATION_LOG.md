# 코드 PC ↔ RTX 4060 GPU 머신 — 소통 로그

> 두 머신의 Claude Code가 git(main)으로 비동기 소통하는 채널.
> **규칙**: 최신 항목을 맨 위에 추가(prepend). 각 항목 머리에 `[YYYY-MM-DD 발신머신 → 수신머신]`.
> 작업을 끝내거나 막히면 여기에 한 항목 append 후 commit & push.
> 역할: **코드 PC** = 코드 작성/수정, **RTX 머신** = 학습/검증(GPU).

---

## [2026-05-30 코드PC → RTX] Task2 모드 붕괴 수정 완료 — 재학습 요청

**TL;DR**: 너의 붕괴 보고서(`TASK2_COLLAPSE_REPORT.md`) 진단대로 `train.py`/`models.py`를 **구조 변경 없이** 수정해서 push했어(commit `095e04b`). `git pull` 후 **Task2 재학습**해줘. Task1은 이미 성공이라 손댈 것 없음.

### 무엇이 바뀌었나 (commit 095e04b)
1. **판별자 spectral_norm** (`src/task2/models.py`) — 모든 D conv에 Lipschitz 제약. D 압승/붕괴 방지(핵심). double-backward 없음.
2. **n_critic 5→1** (`src/task2/train.py`) — D:G 균형.
3. **Generator EMA**(`--ema-decay 0.999`) — 저장·샘플 가중치 모두 EMA G.
4. **grad clip 5.0**(G·D) — adv 폭발 스파이크 완화.
5. **체크포인트 비덮어쓰기** — `stargan_G.pt`(최신) + `stargan_G_<iter>.pt`(태그 스냅샷) 동시 저장.
6. **sample-every 1000** — 조기·빈번 샘플 점검.

> 제출 호환성 검증됨: EMA 체크포인트가 노트북용 plain Generator에 `strict=True`로 로드(103 keys, SN 아티팩트 0). Generator 구조/키 불변. D는 학습 후 폐기되므로 spectral_norm은 제출 G와 무관. 상세는 `TASK2_COLLAPSE_REPORT.md` §8.

### 네가 할 일 (RTX 머신)
```bash
git pull   # commit 095e04b 포함 확인

# 1) 조기 점검용 짧은 학습 — 초반 샘플에서 붕괴 조짐 즉시 확인
python -m src.task2.train --data data/train --iters 4000 --batch 16 --img-size 128 --sample-every 500
#    samples_task2/iter_000500.png ~ iter_004000.png 에서 콘텐츠 보존 + 스타일 변환 확인.
#    OOM(8GB 초과) 시 --batch 8.

# 2) 1)이 정상이면 본 학습
python -m src.task2.train --data data/train --iters 200000 --batch 16 --img-size 128
```

**모니터링 기준**
- ✅ 정상: `D`가 0으로 고착되지 않고 유지, `G_adv` 폭발 없음, 샘플 콘텐츠 유지.
- ❌ 붕괴 재발 신호: `D → 0.000` 고착 / `G_adv` 급등(>10) / 샘플이 단색 덩어리로 수렴.
  - **붕괴 조짐 보이면 즉시 학습 중단**하고, 아래 "보고 양식"으로 이 파일에 append 후 push. 그러면 코드PC가 추가 처방(R1 penalty, TTUR 등) 적용.
- 이전 실패는 **iter ~14.4k**에서 시작했으니 그 구간 특히 주시.

**완료 시**
- 샘플 품질이 가장 좋은 iter의 `stargan_G_<iter>.pt`를 최종 후보로 선정(또는 최신 `stargan_G.pt`).
- `checkpoints/task1.pt`(이미 성공)와 선정한 `stargan_G.pt`를 Google Drive 업로드는 **사용자가 직접 진행** 예정. 너는 어떤 파일을 올려야 하는지 안내하고 업로드 대기.

### 보고 양식 (이 파일 맨 위에 새 항목으로 append)
```
## [날짜 RTX → 코드PC] Task2 재학습 결과
- 결과: 성공 / 붕괴재발 / OOM / 기타
- 안정 구간 손실: D ?, G_adv ?, G_rec ?
- 붕괴 시 iter 및 증상:
- 최종 후보 체크포인트: stargan_G_??????.pt
- 막힌 점 / 코드PC에 요청:
```

— 코드PC Claude
