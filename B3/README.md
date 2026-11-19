# B3 · Cropr FP32 학습 후 PTQ

2-a의 Food-101 실험을 재현한다. ViT-Small과 Cropr query를 FP32로 함께 학습하고, validation Top-1이 가장 높은 `best.pt`에 static affine PTQ를 적용한다. ZIP에 포함된 `best.pt`에서 바로 보정을 시작할 수도 있다.

## 제공된 체크포인트로 실행

저장소 루트와 기존 `vitqat` 환경에서 실행한다. 아래 예시는 ZIP의 `best.pt`와 `classes.json`을 `B3/runs/2-a/`에 둔 경우다. Food-101은 `data/food-101/`에서 읽는다. 처음 내려받을 때는 `--download`를 추가한다.

```powershell
python -m B3.ptq --checkpoint B3/runs/2-a/best.pt --data-root data --output B3/runs/w2a8 --weight-bits 2 --activation-bits 8 --calibration-samples 1024
python -m B3.inference --eval-checkpoint B3/runs/2-a/best.pt --data-root data --output B3/runs/eval-fp32
python -m B3.inference --eval-checkpoint B3/runs/w2a8/ptq.pt --data-root data --output B3/runs/eval-w2a8
```

두 평가 폴더의 `metrics.json`과 `metrics.csv`에서 동일한 공식 test split의 Top-1, loss, 토큰 수, 지연시간을 비교한다. ZIP에 기록된 **77.782178%**는 제공된 FP32 모델의 test Top-1이다. PTQ 지연시간은 PyTorch에서 FP32 커널과 fake quantization을 함께 실행해 측정한다.

## 처음부터 FP32 학습

```powershell
python -m B3.train --data-root data --output B3/runs/fp32
python -m B3.ptq --checkpoint B3/runs/fp32/best.pt --data-root data --output B3/runs/trained-w2a8
```

학습 기본값은 ZIP의 `config.json`을 따른다. 224×224 ViT-Small, pruning rate 16, LLF 활성화, seed 42, 300 epoch, warmup 20 epoch, batch 128이다. Mixup 0.8, CutMix 1.0, Random Erasing 0.25와 가중치 1.0의 보조 분류 손실을 사용한다. 첫 pruning 단계에서 토큰 17개, 이후 각 단계에서 16개를 제거하고 마지막 블록 직전에 197개로 복원한다. 중단된 학습은 같은 명령에 `--resume B3/runs/fp32/last.pt`를 추가해 재개한다.

## PTQ 보정 조건

- B3의 보정 기본값은 W2A8과 최대 1,024장이다. ZIP의 학습 설정에 남은 bit 필드는 FP32 모델의 정밀도에 영향을 주지 않는다. `--weight-bits`와 `--activation-bits`는 각각 2, 3, 4, 8 중에서 선택한다.
- 원본 체크포인트의 split 해시를 검증한 뒤, 해당 seed로 학습 인덱스를 비복원 추출한다. 보정 이미지에는 평가용 resize·center crop·정규화를 적용한다. 보정 중에는 모델 가중치를 고정하고 fake quantization을 끈 채 누적 최솟값·최댓값을 수집한다.
- 가중치는 출력 채널별, activation은 텐서별 affine grid를 쓴다. 기본 `--ptq-scope all`은 backbone의 Conv·Linear·attention과 Cropr의 추론 scorer까지 보정한다. `backbone`을 선택하면 Cropr scorer는 FP32로 계산한다. 학습용 보조 head는 FP32로 보존한다.
- `ptq.pt`에 모델, 고정된 observer, bit 설정, 실제 보정 인덱스와 원본 split 해시를 함께 저장한다. 이후 평가는 이 파일을 복원한다.

## 단일 이미지 추론

```powershell
python -m B3.predict food.jpg --checkpoint B3/runs/w2a8/ptq.pt --topk 5
```

FP32 `best.pt`도 같은 방법으로 사용할 수 있다. FP32 클래스명은 체크포인트 옆의 ZIP `classes.json` 또는 학습 결과 `split.json`에서 읽고, PTQ 클래스명은 `ptq.pt`에서 읽는다.

## 코드와 검증

`train.py`, `ptq.py`, `inference.py`가 각 단계의 진입점이다. `model.py`와 `cropr_adapter.py`가 원본 Cropr 모델을 조립하고, `quantization.py`가 추론 경로의 양자화를 담당한다. 데이터, 학습 및 평가 runner, 체크포인트, 계측은 저장소의 `utils`를 사용한다.

```powershell
python -m pytest B3/tests -q
```
