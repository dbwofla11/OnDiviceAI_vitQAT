# A2: Dense ViT QAT

토큰 수를 유지한 dense ViT에 QAT(Quantization-Aware Training)를 적용한다. A1과 비교해 양자화의 영향을 확인하고, A3에 전달할 QAT-only checkpoint를 만든다. 학습 중 observer와 fake quantization을 사용하며 추론에서는 저장된 양자화 상태를 고정한다.

현재 데이터셋은 Food-101(101 classes), 기본 모델은 `vit_tiny_patch16_224`, 입력 크기는 224다.
비교 실험에서는 모델·초기화(scratch/pretrained)·seed·데이터 split·증강·학습 조건을 맞춘다.

QAT는 FP32 연산에 fake quantization을 적용하는 구현이다. 일반 `inference`의 지연시간은 실제 정수 커널 성능을 의미하지 않는다.

## 폴더 구조

```text
A2/
  __init__.py                 # Python package
  config.py                   # CLI, 실험별 기본값, checkpoint 설정 검증
  model.py                    # 모델 생성, pretrained 초기화, checkpoint 복원
  train.py                    # optimizer, loss, 학습 루프, LR·증강·best 선택 정책
  inference.py                # 저장된 checkpoint의 test split 평가
  metrics.py                  # 측정·결과 항목·환경 정보·model_state.pt 저장
  quantization.py             # 로컬 quantizer, QAT 적용·복원 검증, observer
  tensorrt_int8.py            # W2A8 → INT8 Q/DQ 변환과 ONNX export
  benchmark_tensorrt_int8.py  # FP32 master / 변환 모델의 TensorRT 비교
  README.md                   # 이 실험의 실행 안내
```

`utils`는 공통 기능에만 사용한다. 모델·학습·추론 정책은 이 폴더가 소유한다.

## 실행 준비

아래 명령은 모두 **저장소 루트**에서 실행한다. `cd A2` 후 직접 `train.py`를 실행하지 않는다.
Python 3.10 환경에서 루트의 단일 requirements 파일을 설치한다.

```powershell
python -m pip install -r requirements.txt
```

예시는 CUDA 학습 환경을 사용한다. 일반 학습·추론은 `--device cpu`로도 실행할 수 있다.
`--data-root data`는 Food-101을 저장한 루트 경로이며, 최초 실행의 `--download`는 데이터가 없으면 내려받는다.

## 학습

다음은 scratch 학습 예시다. 새 학습의 `--output`은 없거나 비어 있는 디렉터리여야 한다.

```powershell
python -m A2.train --data-root data --download --output A2/runs/w2a8 --device cuda --epochs 300 --warmup-epochs 20 --batch-size 128 --seed 42 --weight-bits 2 --activation-bits 8
```

학습은 매 epoch validation을 수행하고 Top-1이 가장 높은 checkpoint를 `best.pt`로 선택한다.
학습 종료 후에는 `best.pt`를 다시 불러와 test split을 평가한다.

### Pretrained에서 새 학습 시작

```powershell
python -m A2.train --data-root data --output A2/runs/w2a8_pretrained --device cuda --pretrained --model vit_tiny_patch16_224.augreg_in21k_ft_in1k --weight-bits 2 --activation-bits 8
```

`--pretrained`는 새 학습의 backbone을 timm 사전학습 가중치로 초기화한다. Food-101용 head를 사용하며 기본값은 scratch다.
처음에는 사전학습 가중치를 내려받을 수 있다. 재개와 추론에서는 저장된 state를 복원하므로 다시 내려받지 않는다.

### 중단된 학습 재개

아래는 앞의 scratch 실행을 재개하는 예시다.

```powershell
python -m A2.train --data-root data --output A2/runs/w2a8 --resume A2/runs/w2a8/last.pt --device cuda --epochs 300 --warmup-epochs 20 --batch-size 128 --seed 42 --weight-bits 2 --activation-bits 8
```

같은 output의 `last.pt`와 `best.pt`를 함께 보존한다. 모델·epochs·batch size·seed·증강 등 원래 학습 설정을 그대로 전달한다.
Pretrained 실행을 재개한다면 해당 output/last.pt를 지정하고 원래 `--pretrained --model ...` 설정도 유지한다.

## 추론·평가

```powershell
python -m A2.inference --eval-checkpoint A2/runs/w2a8/best.pt --data-root data --output A2/runs/eval --device cuda --batch-size 128 --workers 4 --latency-warmup 10 --latency-repeats 100
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

## ONNX export와 TensorRT 평가

학습·보정된 **W2A8 checkpoint**만 현재 converter에 입력할 수 있다. 위 학습 예시가 이 조건을 사용한다.
`--export-only`는 CPU에서 ONNX와 변환 점검 결과를 생성하며, TensorRT 엔진 실행과 실제 데이터 정확도 평가를 생략한다.

```powershell
python -m A2.benchmark_tensorrt_int8 --checkpoint A2/runs/w2a8/best.pt --output A2/runs/onnx --export-only
```

다음은 CUDA에서 엔진을 생성하고 실제 test split 정확도와 지연시간을 측정한다.

```powershell
python -m A2.benchmark_tensorrt_int8 --checkpoint A2/runs/w2a8/best.pt --output A2/runs/tensorrt --device cuda --data-root data --eval-split test --eval-batch-size 32 --workers 4 --warmup 20 --iterations 100
```

- `--export-only`와 `--data-root`는 함께 쓰지 않는다. Native 실행에서 `--data-root`를 생략하면 실제 데이터 정확도는 평가하지 않는다.
- Timing batch는 1로 고정된다. `--eval-batch-size`는 정확도 평가용 batch이며 timing batch와 별개다.
- `fp32_master`는 동일 QAT checkpoint의 FP32 master weights다. 별도로 학습한 A1 baseline이 아니다.
- W2 값을 INT8 Q/DQ로 표현하고 A8 grid를 재매핑한다. INT2 커널을 실행한다는 의미가 아니며 실제 커널 정밀도는 layer report로 확인한다.
- export에는 ONNX, native 실행에는 CUDA와 TensorRT 10이 필요하다. 의존성은 루트 `requirements.txt`에서 관리한다.

Export 결과는 `fp32_master.onnx`, `w2_int8.onnx`, `report.json`에 저장한다. Native 실행은 engine·layer/timing 결과와 `metrics.csv`도 생성한다.

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
python -m A2.train --help
python -m A2.inference --help
python -m A2.benchmark_tensorrt_int8 --help
python -m pytest tests/test_experiment_entrypoints.py tests/test_tensorrt_int8.py -q
```

[전체 실험 구조와 공통 코드 경계](../docs/experiment-folder-structure.md), [학습 조건·지표 정의](../docs/experiments.md).
