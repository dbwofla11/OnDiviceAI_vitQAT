# A1: FP32 dense baseline

Food-101에서 토큰을 제거하지 않는 FP32 ViT 기준선을 학습한다. A2의 양자화 효과와 B1/B2의 learned-query pruning + QAT 효과를 비교하는 기준이다. A1 checkpoint는 B1의 `--init-checkpoint` FP32 초기값으로도 쓸 수 있다. 학습·추론 모두 dense forward를 사용하며 QAT와 pruning은 적용하지 않는다.

`--device cuda`로 평가하면 QAT 실험의 INT8 엔진과 같은 백엔드로 비교할 수 있도록, 공용 함수(`utils/tensorrt.py`)가 FP32 TensorRT 엔진 시간을 `tensorrt_fp32_latency_ms*`에 기록한다(TensorRT가 없으면 `null`과 이유).

현재 데이터셋은 Food-101(101 classes), 기본 모델은 `vit_tiny_patch16_224`, 입력 크기는 224다.
비교 실험에서는 모델·초기화(scratch/pretrained)·seed·데이터 split·증강·학습 조건을 맞춘다.

## 폴더 구조

```text
A1/
  __init__.py   # Python package
  config.py     # CLI, 실험별 기본값, checkpoint 설정 검증
  model.py      # 모델 생성, pretrained 초기화, checkpoint 복원
  train.py      # optimizer, loss, 학습 루프, LR·증강·best 선택 정책
  inference.py  # 저장된 checkpoint의 test split 평가
  metrics.py    # 측정·결과 항목·환경 정보·model_state.pt 저장
  README.md     # 이 실험의 실행 안내
```

`utils`는 공통 기능에만 사용한다. 모델·학습·추론 정책은 이 폴더가 소유한다.

## 실행 준비

아래 명령은 모두 **저장소 루트**에서 실행한다. `cd A1` 후 직접 `train.py`를 실행하지 않는다.
Python 3.10 환경에서 루트의 단일 requirements 파일을 설치한다.

```powershell
python -m pip install -r requirements.txt
```

예시는 CUDA 학습 환경을 사용한다. 일반 학습·추론은 `--device cpu`로도 실행할 수 있다.
`--data-root data`는 Food-101을 저장한 루트 경로이며, 최초 실행의 `--download`는 데이터가 없으면 내려받는다.

## 학습

다음은 scratch 학습 예시다. 새 학습의 `--output`은 없거나 비어 있는 디렉터리여야 한다.

```powershell
python -m A1.train --data-root data --download --output A1/runs/fp32 --device cuda --epochs 300 --warmup-epochs 20 --batch-size 128 --seed 42
```

학습은 매 epoch validation을 수행하고 Top-1이 가장 높은 checkpoint를 `best.pt`로 선택한다.
학습 종료 후에는 `best.pt`를 다시 불러와 test split을 평가한다.

### Pretrained에서 새 학습 시작

```powershell
python -m A1.train --data-root data --output A1/runs/fp32_pretrained --device cuda --pretrained --model vit_tiny_patch16_224.augreg_in21k_ft_in1k
```

`--pretrained`는 새 학습의 backbone을 timm 사전학습 가중치로 초기화한다. Food-101용 head를 사용하며 기본값은 scratch다.
처음에는 사전학습 가중치를 내려받을 수 있다. 재개와 추론에서는 저장된 state를 복원하므로 다시 내려받지 않는다.

### 중단된 학습 재개

아래는 앞의 scratch 실행을 재개하는 예시다.

```powershell
python -m A1.train --data-root data --output A1/runs/fp32 --resume A1/runs/fp32/last.pt --device cuda --epochs 300 --warmup-epochs 20 --batch-size 128 --seed 42
```

같은 output의 `last.pt`와 `best.pt`를 함께 보존한다. 모델·epochs·batch size·seed·증강 등 원래 학습 설정을 그대로 전달한다.
Pretrained 실행을 재개한다면 해당 output/last.pt를 지정하고 원래 `--pretrained --model ...` 설정도 유지한다.

## 추론·평가

```powershell
python -m A1.inference --eval-checkpoint A1/runs/fp32/best.pt --data-root data --output A1/runs/eval --device cuda --batch-size 128 --workers 4 --latency-warmup 10 --latency-repeats 100
```

이 명령은 Food-101 test split 전체를 평가한다. 모델 구조·초기화 이력·전처리 설정은 checkpoint에서 읽으며,
추론 명령에 `--model`, `--pretrained`, `--epochs`를 다시 넘기지 않는다. 데이터 경로와 실행 device·batch size는 변경할 수 있으나 원래 데이터 split은 일치해야 한다.

## 주요 옵션

모델·학습 정책 옵션은 `train`에서 사용한다. `inference`는 checkpoint 경로와 데이터·실행 환경 옵션을 받으며, 정확한 목록은 각 명령의 `--help`로 확인한다.
실험별 FP32/QAT·pruning 방식은 고정돼 있으므로 `--mode`를 따로 지정할 필요가 없다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model`, `--image-size` | `vit_tiny_patch16_224`, `224` | 학습 모델과 입력 크기 |
| `--epochs`, `--warmup-epochs` | `300`, `20` | 학습 기간과 LR warmup. 짧게 학습할 때 warmup은 epochs보다 작게 설정 |
| `--lr`, `--min-lr`, `--weight-decay` | `5e-4`, `1e-6`, `0.05` | AdamW와 cosine LR 설정 |
| `--batch-size`, `--accum-steps` | `128`, `1` | batch 크기와 학습 gradient accumulation 횟수 |
| `--seed`, `--val-fraction` | `42`, `0.1` | 난수 seed와 공식 train에서 나누는 validation 비율 |
| `--mixup`, `--cutmix`, `--random-erasing` | 모두 `0` | 추가 증강 강도/확률. 기본 crop·flip·RandAugment는 별도로 적용 |
| `--workers`, `--threads` | `4`, `0` | DataLoader worker와 CPU thread 수. threads=0은 기존 설정 유지 |
| `--latency-warmup`, `--latency-repeats` | `10`, `100` | 일반 평가의 batch=1 forward 지연시간 측정 횟수 |

## 출력 파일

| 파일 | 내용 |
|---|---|
| `config.json`, `split.json` | 실행 설정과 데이터 split 정보 |
| `epochs.jsonl` | epoch별 train/validation 지표(학습 시 생성) |
| `last.pt` | 마지막 epoch의 모델·optimizer·RNG·설정, 재개용 |
| `best.pt` | 최고 validation Top-1 checkpoint, 평가용 |
| `metrics.json`, `metrics.csv` | test 정확도·loss·모델 크기·연산량·지연시간 |
| `model_state.pt` | 선택된 모델의 state dict만 저장한 파일 |

별도 inference output에는 config/split/metrics/model_state를 저장한다. `--eval-checkpoint`에는 설정이 포함된
`best.pt` 또는 `last.pt`를 사용한다. `model_state.pt`만으로는 모델 설정과 데이터 split을 복원할 수 없다.

## Pretrained·새 모델 구현 위치

`model.py`의 `build_backbone`/`build_model`/`restore_model`에서 가중치 초기화와 복원을 변경하고,
필요한 CLI 옵션은 `config.py`에 추가한다. 전처리가 다르면 로컬 데이터 함수를 만들어
`train.py`·`inference.py`에 연결하고, 측정 규약이 다르면 `metrics.py`를 수정한다.
새 모델을 지원하기 위해 다른 실험이나 `utils`에 모델 선택 분기를 추가하지 않는다.

## 도움말과 검증

```powershell
python -m A1.train --help
python -m A1.inference --help
python -m pytest tests/test_experiment_entrypoints.py -q
```

[전체 실험 구조와 공통 코드 경계](../docs/experiment-folder-structure.md), [학습 조건·지표 정의](../docs/experiments.md).
