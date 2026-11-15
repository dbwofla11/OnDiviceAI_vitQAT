# A3: QAT checkpoint의 추론 전용 CLS attention pruning

A2에서 학습한 QAT-only checkpoint 하나로 dense 추론과 CLS attention Top-K pruning 추론을 비교한다.
두 조건은 동일한 backbone 가중치·양자화 상태·데이터 split을 사용한다. 정확도 변화, FLOPs, 토큰 수,
지연시간을 비교하여 **추가 학습 없이 추론 시 pruning을 적용한 영향**을 확인한다.

**A3에는 `train.py`가 없다.** 학습은 A2에서 수행하며, A3는 checkpoint를 복원해 평가만 한다.
학습 가능한 query나 보조 head를 추가하지 않는다. 학습부터 CLS pruning을 적용하려면 [A4](../A4/README.md)를 사용한다.

## 폴더 구조

```text
A3/
  __init__.py                 # Python package
  model.py                    # 선행 dense ViT/QAT 모델 생성과 checkpoint 복원
  quantization.py             # A3가 소유하는 QAT 연산·observer·복원 검증
  pruning/                    # CLS attention scorer, 토큰 제거 스케줄, wrapper
  inference.py                # 권장 paired 평가 CLI
  eval_pruning.py             # CLI·설정 상속·dense/pruned 평가·결과 저장
  inference_pruning.py        # 기존 Python import 경로 호환용
  metrics.py                  # 토큰·연산량·QAT 저장 비용·지연시간 측정
  tensorrt_int8.py            # W2A8 INT8 Q/DQ 변환과 ONNX export
  benchmark_tensorrt_int8.py  # dense_int8 / a3_int8 export와 native 비교
  tests/                      # paired 평가·pruning·export 회귀 테스트
  README.md                   # 실행 안내
```

A2의 Python 구현을 import하지 않고 A3의 모델·quantizer로 A2 checkpoint를 복원한다.
A2의 저장 표현을 변경할 때는 A3의 복원 호환성도 확인해야 한다.

## 실험 동작

각 pruning 단계에서 여러 attention head의 CLS attention을 평균해 토큰 순위를 정한다.
CLS 토큰은 유지하고 나머지 patch token 중 높은 점수의 토큰을 남긴다.
현재 CLI의 스케줄은 `cropr-r16-first17-llf` 하나로 고정돼 있다.

- 첫 단계에서 17개, 이후 단계에서 16개씩 토큰을 제거한다.
- 마지막 block 직전에는 앞에서 제거한 토큰을 다시 합친다(LLF).
- 기본 224 입력·patch 16·12-block ViT의 block별 입력 토큰 수는 다음과 같다.

```text
dense:  197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197
pruned: 197, 180, 164, 148, 132, 116, 100,  84,  68,  52,  36, 197
```

QAT observer는 고정한다. 평가 전후 모델 state와 입력 checkpoint의 hash가 유지되는지도 확인한다.
일반 추론은 fake quantization을 사용하는 FP32 연산이며, 정수 커널 성능은 별도의 TensorRT 명령으로 측정한다.

## 실행 준비

모든 명령은 **저장소 루트**에서 실행한다. Python 3.10 환경에서 단일 requirements 파일을 설치한다.

```powershell
python -m pip install -r requirements.txt
```

입력은 A2의 설정·모델 state·데이터 split 정보가 포함된 checkpoint여야 한다.
A1 FP32, A4 pruning 학습, B2 learned-query Cropr checkpoint는 사용할 수 없다.
`model_state.pt`만 전달하는 것도 지원하지 않는다.

아직 A2 checkpoint가 없다면 다음과 같이 준비한다. 이 예시는 이후 TensorRT 평가에도 사용할 수 있는 W2A8이다.
이미 학습한 A2 checkpoint가 있으면 이 단계를 생략한다.

```powershell
python -m A2.train --data-root data --download --output A2/runs/w2a8 --device cuda --weight-bits 2 --activation-bits 8
```

A2를 pretrained에서 시작하려면 A2 학습 명령에
`--pretrained --model vit_tiny_patch16_224.augreg_in21k_ft_in1k`를 추가하고 새 output을 지정한다.
A3에는 `--pretrained` 옵션이 없으며, A2에서 학습한 pretrained 기반 checkpoint를 그대로 사용한다.
A3는 추가 가중치를 다운로드하거나 학습하지 않는다.

## Dense / pruned paired 평가

```powershell
python -m A3.inference --checkpoint A2/runs/w2a8/best.pt --data-root data --output A3/runs/paired --device cuda --workers 4 --eval-batch-size 128 --latency-warmup 10 --latency-repeats 100
```

한 번의 실행으로 Food-101 test split 전체를 두 조건에서 평가한다. `--output`은 새 디렉터리 또는 빈 디렉터리여야 한다.
학습 시의 모델·입력 크기·bit·seed·split 설정은 checkpoint에서 상속한다.
일반 paired 평가는 저장된 2·3·4·8bit QAT 설정을 지원하며 W2A8로 제한하지 않는다.

`--device cpu`로도 실행할 수 있다. A3에는 `--download`가 없으므로 A2에서 사용한 Food-101 데이터를 미리 준비한다.
입력 크기나 모델 깊이를 바꾼 checkpoint는 현재 고정 pruning 스케줄로 최소 한 개의 patch token을 남길 수 있어야 한다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--checkpoint` | 필수 | A2의 `best.pt` 등 config와 모델 state가 포함된 QAT checkpoint |
| `--output` | 필수 | 새 paired 평가 결과 경로 |
| `--data-root` | checkpoint 값 | 데이터 위치만 변경 가능. 원래 split과 일치해야 함 |
| `--evaluation-id` | `S-QAT-P` | 결과를 구별하는 평가 label |
| `--eval-batch-size` | checkpoint의 batch size | dense/pruned에 동일하게 적용할 정확도 평가 batch |
| `--workers`, `--threads` | checkpoint 값(threads 누락 시 0) | DataLoader worker와 CPU thread 설정 재정의 |
| `--latency-warmup`, `--latency-repeats` | `10`, `100` | 동일 validation 샘플을 이용한 batch=1 timing 횟수 |
| `--pruning-schedule` | `cropr-r16-first17-llf` | 현재 지원하는 단일 스케줄 |

`python -m A3.eval_pruning`도 같은 CLI를 제공한다. 새 실행에는 `A3.inference`를 사용한다.

## ONNX export와 TensorRT 비교

이 경로는 **학습·보정된 A2 W2A8 checkpoint**만 지원한다.
CPU에서 두 조건의 ONNX를 생성하고 변환 결과를 점검하려면 다음을 실행한다.

```powershell
python -m A3.benchmark_tensorrt_int8 --checkpoint A2/runs/w2a8/best.pt --output A3/runs/onnx --export-only
```

`--export-only`는 데이터셋 평가와 TensorRT 실행을 생략한다. ONNX 의존성이 필요하다.
다음 명령은 CUDA/TensorRT 10에서 두 엔진의 정확도와 지연시간을 비교한다.

```powershell
python -m A3.benchmark_tensorrt_int8 --checkpoint A2/runs/w2a8/best.pt --data-root data --output A3/runs/tensorrt --device cuda --eval-split test --eval-batch-size 32 --workers 4 --threads 2 --warmup 20 --iterations 100
```

- `dense_int8`는 pruning 없는 변환 모델, `a3_int8`는 동일 모델에 CLS pruning을 적용한 조건이다.
- Native 실행은 실제 데이터를 평가한다. `--data-root`를 생략하면 checkpoint의 데이터 경로를 사용한다.
- Timing batch는 1로 고정되며, `--eval-batch-size`는 정확도 평가용 batch다.
- `--warmup`, `--iterations`는 TensorRT timing 횟수다. 일반 추론의 `--latency-*`와 옵션 이름이 다르다.
- W2 값을 INT8 Q/DQ로 표현하고 A8 grid를 재매핑한다. 실제 커널 정밀도는 layer report로 확인한다.
- 일반 paired 평가와 마찬가지로 output은 새 경로 또는 빈 디렉터리여야 한다.

## 출력 파일

일반 paired 평가에서는 다음 파일을 생성한다. 학습·재개 checkpoint는 생성하지 않는다.

| 파일 | 내용 |
|---|---|
| `evaluation_config.json` | 원본 학습 설정, 평가 옵션, 유효 설정, pruning 스케줄 |
| `split.json` | 원본과 일치하는 데이터 split |
| `dense_metrics.json`, `pruned_metrics.json` | 각 조건의 정확도·loss·토큰 수·연산량·지연시간 |
| `comparison.json` | 정확도 차이, latency/FLOPs 비율, state 유지 여부 |
| `comparison.csv` | dense/pruned 두 조건의 지표 |

TensorRT 경로는 `dense_int8.onnx`, `a3_int8.onnx`, `report.json`, `metrics.csv`를 생성한다.
Native 실행 시에는 engine·layer·timing 결과도 저장한다.

## 모델·추론 확장 위치

선행 모델의 생성·복원 표현은 `model.py`와 `quantization.py`, pruning 방식은 `pruning/`,
설정 상속과 paired 평가 절차는 `eval_pruning.py`, 측정 항목은 `metrics.py`에서 관리한다.
전처리가 다르면 로컬 데이터 함수를 연결한다. A3의 변경을 위해 다른 실험이나 utils에 모델 분기를 추가하지 않는다.

## 도움말과 검증

```powershell
python -m A3.inference --help
python -m A3.benchmark_tensorrt_int8 --help
python -m pytest A3/tests -q
```

[CLS attention pruning 설명](../docs/A3/cls-attention-pruning.md),
[전체 실험 구조와 코드 소유권](../docs/experiment-folder-structure.md).
