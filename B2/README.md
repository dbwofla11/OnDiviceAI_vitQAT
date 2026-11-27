# B2: Learned-query Cropr + QAT

learned-query Cropr와 QAT를 함께 학습한다. 주 classification loss와 보조 classification loss로 backbone·query·보조 head를 학습하며, 추론에서는 토큰 pruning과 저장된 fake-quant 상태를 사용한다.
B1(Cropr + Linear QAT, LSQ/LSQ+)과 비교하면 같은 Cropr 구조에서 양자화 방식과 범위의 영향을 확인할 수 있다.
B2는 PyTorch affine observer로 모든 Linear·attention Q/K/V·확률을 양자화한다.

현재 데이터셋은 Food-101(101 classes), 기본 모델은 `vit_tiny_patch16_224`, 입력 크기는 224다.
비교 실험에서는 모델·초기화(scratch/pretrained)·seed·데이터 split·증강·학습 조건을 맞춘다.

QAT는 FP32 연산에 fake quantization을 적용하는 구현이다. 학습과 `inference`는 같은 quantizer를 사용하므로 양자화 방식이 같다.
둘 다 비대칭(affine) 양자화이며 weight는 출력 채널별, activation은 tensor별로 `0..2^b-1` 범위와 zero point를 쓴다.
Cropr scorer도 학습(`forward`)과 추론(`forward_scorer`)이 같은 quantizer를 공유한다.
`latency_ms_*`는 fake quantization(FP32 연산) 시간이라 정수 커널 성능이 아니다. `--device cuda`로 평가하면 공용 함수(`utils/int8.py`, `utils/tensorrt.py`)가 같은 grid의 INT8 Q/DQ 복사본을 TensorRT 엔진으로 실행해 `integer_latency_ms*`에 기록한다(TensorRT가 없으면 `null`과 이유). Cropr scorer도 INT8 곱으로 실행된다(`B2/int8.py`).

## 폴더 구조

```text
B2/
  __init__.py                 # Python package
  config.py                   # CLI, 실험별 기본값, checkpoint 설정 검증
  model.py                    # 모델 생성, pretrained 초기화, checkpoint 복원
  train.py                    # optimizer, loss, 학습 루프, LR·증강·best 선택 정책
  inference.py                # 저장된 checkpoint의 test split 평가
  metrics.py                  # 측정·결과 항목·환경 정보·model_state.pt 저장
  cropr_adapter.py            # 이 실험의 원본 Cropr 조립·가중치 전달
  quantization.py             # backbone·CrossAttention QAT, observer, 복원 검증
  tests/                      # query·보조 loss·QAT scope·checkpoint 검증
  README.md                   # 이 실험의 실행 안내
```

`utils`는 공통 기능에만 사용한다. 모델·학습·추론 정책은 이 폴더가 소유한다.

## 실행 준비

아래 명령은 모두 **저장소 루트**에서 실행한다. `cd B2` 후 직접 `train.py`를 실행하지 않는다.
Python 3.10 환경에서 루트의 단일 requirements 파일을 설치한다.

```powershell
python -m pip install -r requirements.txt
```

예시는 CUDA 학습 환경을 사용한다. 일반 학습·추론은 `--device cpu`로도 실행할 수 있다.
`--data-root data`는 Food-101을 저장한 루트 경로이며, 최초 실행의 `--download`는 데이터가 없으면 내려받는다.

## 학습

다음은 scratch 학습 예시다. 새 학습의 `--output`은 없거나 비어 있는 디렉터리여야 한다.

```powershell
python -m B2.train --data-root data --download --output B2/runs/cropr_w2a8 --device cuda --epochs 300 --warmup-epochs 20 --batch-size 128 --seed 42 --weight-bits 2 --activation-bits 8 --cropr-pruning-rate 8 --aux-loss-weight 1.0 --qat-scope all
```

학습은 매 epoch validation을 수행하고 Top-1이 가장 높은 checkpoint를 `best.pt`로 선택한다.
학습 종료 후에는 `best.pt`를 다시 불러와 test split을 평가한다.

### Pretrained에서 새 학습 시작

```powershell
python -m B2.train --data-root data --output B2/runs/cropr_w2a8_pretrained --device cuda --pretrained --model vit_tiny_patch16_224.augreg_in21k_ft_in1k --weight-bits 2 --activation-bits 8 --cropr-pruning-rate 8 --aux-loss-weight 1.0 --qat-scope all
```

`--pretrained`는 새 학습의 backbone을 timm 사전학습 가중치로 초기화한다. Food-101용 head를 사용하며 기본값은 scratch다.
처음에는 사전학습 가중치를 내려받을 수 있다. 재개와 추론에서는 저장된 state를 복원하므로 다시 내려받지 않는다.
Cropr query와 보조 head는 이 실험에서 학습한다. 현재 adapter와 호환되는 ViT(단일 CLS, MLP ratio 4 등)를 사용해야 한다.

### 중단된 학습 재개

아래는 앞의 scratch 실행을 재개하는 예시다.

```powershell
python -m B2.train --data-root data --output B2/runs/cropr_w2a8 --resume B2/runs/cropr_w2a8/last.pt --device cuda --epochs 300 --warmup-epochs 20 --batch-size 128 --seed 42 --weight-bits 2 --activation-bits 8 --cropr-pruning-rate 8 --aux-loss-weight 1.0 --qat-scope all
```

같은 output의 `last.pt`와 `best.pt`를 함께 보존한다. 모델·epochs·batch size·seed·증강 등 원래 학습 설정을 그대로 전달한다.
Pretrained 실행을 재개한다면 해당 output/last.pt를 지정하고 원래 `--pretrained --model ...` 설정도 유지한다.

## 추론·평가

```powershell
python -m B2.inference --eval-checkpoint B2/runs/cropr_w2a8/best.pt --data-root data --output B2/runs/eval --device cuda --batch-size 128 --workers 4 --latency-warmup 10 --latency-repeats 100
```

이 명령은 Food-101 test split 전체를 평가한다. 모델 구조·초기화 이력·전처리 설정은 checkpoint에서 읽으며,
추론 명령에 `--model`, `--pretrained`, `--epochs`를 다시 넘기지 않는다. 데이터 경로와 실행 device·batch size는 변경할 수 있으나 원래 데이터 split은 일치해야 한다.
양자화 bit와 scope도 checkpoint에서 복원한다. 비트 설정 불일치나 미보정·손상된 observer는 복원 시 거부한다.

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
| `--weight-bits`, `--activation-bits` | 각각 `2` | 각각 2·3·4·8bit 지원. 기본 W2A2, 위 예시는 W2A8 |
| `--freeze-observers-epoch` | `-1` | -1이면 학습 내내 observer 갱신. 1 이상이면 해당 0-based epoch부터 고정 |
| `--cropr-pruning-rate` | `8` | 기본 제거 토큰 수. 첫 단계는 이 값보다 1개 더 제거 |
| `--cropr-llf` / `--no-cropr-llf` | 비활성 | 마지막 block 전에 제거한 토큰을 다시 합치는 LLF 사용 여부 |
| `--aux-loss-weight` | `1.0` | 보조 classification loss 합에 곱하는 양수 계수 |
| `--qat-scope` | `all` | `all`: Cropr까지 QAT, `backbone`: `cropr.*` 모듈은 FP32 유지 |

입력 크기·patch 크기·block 수를 바꾸면 제거 가능한 토큰 수도 달라진다. 최소 한 개의 patch token이 남도록 pruning rate를 조정한다.

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
python -m B2.train --help
python -m B2.inference --help
python -m pytest B2/tests -q
```

[전체 실험 구조와 공통 코드 경계](../docs/experiment-folder-structure.md), [학습 조건·지표 정의](../docs/experiments.md).
