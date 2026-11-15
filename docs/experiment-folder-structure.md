# 실험 코드의 소유권과 실행 구조

각 실험은 자체 모델·학습·추론·양자화 구현을 소유한다. 공통 runner에 mode를 넘겨
알고리즘을 고르는 방식은 제거했다. 다른 실험을 import해서 실행하지 않는다.

| 실험 | 학습 | 추론 | 양자화 구현 |
|---|---|---|---|
| A1 | FP32 dense | FP32 checkpoint 평가 | 없음 |
| A2 | QAT-only | QAT checkpoint 평가 | `A2/quantization.py` |
| A3 | 선행 A2 학습 결과 사용 | 동일 checkpoint의 dense/CLS pruning 비교 | `A3/quantization.py` |
| A4 | CLS pruning + QAT | 학습 때의 CLS wrapper 복원 | `A4/quantization.py` |
| B1 | FP32 Cropr | FP32 Cropr 평가 | PTQ 미구현 |
| B2 | learned-query Cropr + QAT | 보조 head 없이 Cropr 평가 | `B2/quantization.py` |

## 각 학습 실험의 파일

A1/A2/A4/B1/B2는 다음 구조를 사용한다.

```text
<실험>/
  config.py        # 학습 인자·기본값, 추론 CLI, 구형 checkpoint 기본값과 검증
  model.py         # timm 모델 생성, scratch/pretrained, QAT/pruning 조립
  train.py         # 학습 루프, optimizer, criterion/loss, Mixup, LR, clipping, observer
  inference.py     # checkpoint 평가 진입점, eval/observer 정책
  metrics.py       # 실험별 측정·결과 항목·환경 메타데이터·모델 파일 저장
  quantization.py  # QAT 실험만: quantizer, QAT 연산, observer, 저장 비용
  cropr_adapter.py # B1/B2: 각 실험이 소유하는 원본 Cropr 모델 조립
  pruning/         # A4: CLS scorer·스케줄·wrapper
```

A3는 `model.py`, `quantization.py`, `pruning/`, `metrics.py`로 선행 모델 복원과
추론 정책을 직접 관리하고, `inference.py`가 `eval_pruning.py`의 paired 평가를 실행한다.
A3/A4의 pruning과 A2/A3/A4/B2의 quantization은 현재 수학적으로 같은 구현에서 출발하지만
파일과 클래스는 실험별로 독립적이다. 한 실험의 변경이 다른 실험으로 자동 전파되지 않는다.
앞으로 A2의 양자화 표현을 바꾸면 A3의 checkpoint 인수 계약도 명시적으로 검토해야 한다.

A2/A3/B2의 `tensorrt_int8.py`와 `benchmark_tensorrt_int8.py`는 각 실험의 export 정책이다.
QAT는 weight/activation 각각 2·3·4·8bit를 지원하고, 현재 TensorRT 변환은 W2A8만 지원한다.

## utils에 남긴 기능

전체 소스 파일을 확인한 결과다. 실행 라이브러리 의존성과 명시적으로 선택하는 공통 기능은
유지한다. 특정 실험을 import하거나 공통 실행기가 실험 정책을 암묵적으로 선택하는 의존성은 제거한다.

| 모듈 | 직접 의존성 | 책임과 명시적 사용 조건 |
|---|---|---|
| `__init__.py` | 없음 | 설명만 제공; 하위 모듈을 자동 import하지 않음 |
| `cli.py` | argparse, torch | 공통 실행 인자와 CUDA 가용성 확인 |
| `constants.py` | 없음 | Food-101의 클래스 수; 모든 모델의 보편적 상수가 아님 |
| `data.py` | numpy, torch, torchvision, augment/constants | 명시적인 `make_food101_data` recipe 및 split/loader 도구 |
| `augment.py` | timm Mixup, torchvision | 공통 이미지 증강. `build_mixup`의 `num_classes`는 호출자가 지정 |
| `classification.py` | torch | 명시적으로 선택하는 CE/Top-1, fixed-batch 분류 평가, logits 비교 |
| `checkpoint.py` | torch, reproducibility | PyTorch 상태 I/O, config/split/RNG 호환 검사; 모델 복원은 callback |
| `reproducibility.py` | numpy, torch | `seed_all`은 RNG만 변경; `configure_backend`는 전달받은 결정성/TF32 설정 적용 |
| `optim.py` | 외부 import 없음 | 호출자가 선택하는 weight-decay 그룹 분리 |
| `runner.py` | torch, checkpoint/reproducibility | 모델·optimizer·epoch·평가·데이터·loader·선택 점수·보고 callback 실행 |
| `evaluation.py` | torch, checkpoint/reproducibility | config/state 복원 및 명시적으로 전달받은 데이터·loader·평가·보고 실행 |
| `results.py` | csv, json, pathlib | 호출자 dict의 JSON/CSV 저장만 수행 |
| `metrics.py` | torch, 표준 라이브러리 | ViT `blocks/embed_dim` 규약의 연산·토큰 수와 지연시간 측정; 정밀도/packed storage를 판단하지 않음 |
| `tensorrt_benchmark.py` | torch, 호출 시 TensorRT | TensorRT 10 고정 shape FP32 images/logits I/O 실행; export·평가·출력 비교·TF32 설정은 호출자가 지정 |

`utils/models.py`, `utils/train.py`, `utils/training.py`, `utils/quantization.py`, `utils/pruning/`,
`utils/pruning_metrics.py`, `utils/tensorrt_int8.py`, `utils/cropr_adapter.py`는 제거했다.
utils에는 실험 선택표, QAT 클래스 import, optimizer 생성, pretrained=False 고정 정책이 없다.
실험 구현은 utils를 사용할 수 있지만 utils는 실험을 import할 수 없다.
`tests/support.py`와 `tests/legacy_config.py`는 이전 다중 실험 회귀 테스트의 입력 어댑터이며
운영 코드에서 사용하지 않는다.

A1/A2/A3/A4는 timm ViT를 직접 사용한다. CLS pruning도 timm의 embedding/head 규약을
사용하므로 일반 train/inference/export 경로에서 `cropr`를 import하지 않는다.
A 계열의 quantization/export 파일에는 learned-query CrossAttention 구현이 없다.
기존 A3/A4 `attach_pruning(..., apply_during="cropr_train_and_inference")` 호환 API만
명시적으로 호출했을 때 원본 `cropr.cls.cropr`를 지연 import한다. 이 경로는 현재 A 계열 실험에서 사용하지 않는다.

`results.py`는 model, args, checkpoint, split 구조를 받지 않으며 torch/timm/torchvision을
import하지 않는다. `save_results(metrics, output)`은 호출자가 정한 항목만 기록한다.
A1/A2/A4/B1/B2의 `metrics.py`가 `report_results`를 제공하여 Top-1/loss, 선택 epoch,
실행 환경 및 `model_state.pt` 저장 방식을 결정한다. 학습·평가 runner는 이 함수를 필수
callback으로 받는다. 다른 실험은 결과 항목이나 artifact 형식을 바꾸기 위해 utils를 수정할 필요가 없다.

`runner/evaluation`은 Food-101 데이터 또는 loader를 기본 선택하지 않는다. 각 실험이
`data_factory`, `loader_factory`를 전달한다. 현재 실험은 공통 Food-101 recipe를 선택하지만
다른 데이터셋을 사용하는 실험은 자체 factory를 전달할 수 있다.
checkpoint 선택도 실험의 `selection_score(validation)`가 결정한다(높을수록 좋음).
현재 실험은 Top-1을 반환하며, 회귀 실험은 음수 MSE 등을 반환할 수 있다.
새 checkpoint는 `best_score`를 저장하고 기존 `best_acc` checkpoint도 복원한다.

공통 ViT 측정 함수는 raw parameter bytes·FLOPs·token trace·latency를 반환한다.
`latency_backend`, quantized parameter 수, packed storage 추정과 pruning 비율 정의는
각 실험의 `metrics.py`가 구성한다. TensorRT의 데이터 선택은 각 benchmark 진입점에서,
CE/Top-1 평가는 명시적으로 전달한 classification callback에서 수행한다.

경계 검증은 `tests/test_utils_boundaries.py`에서 수행한다. timm/torchvision/cropr import를
차단한 상태의 인프라 import, 3-class Mixup의 홀수/짝수 batch, 낮은 MSE 기준 checkpoint
선택·재개, 구형 checkpoint 복원, TF32 설정 보존, 분류 지표 없는 backend 결과 전달을 확인한다.
이는 PyTorch·Food-101·ViT·TensorRT 기능 자체에 대한 의존성이 전혀 없다는 뜻은 아니다.
위 표에 적은 공통 기능의 사용 조건과 backend 지원 범위는 유지한다.

## 실행

```powershell
# scratch QAT 학습
python -m A2.train --data-root data --output A2/runs/scratch

# pretrained ViT로 시작하는 새 학습 (모델이 제공하는 가중치 다운로드 가능)
python -m A2.train --pretrained --model vit_tiny_patch16_224.augreg_in21k_ft_in1k --data-root data --output A2/runs/pretrained

# 저장된 설정·가중치 복원. pretrained를 다시 다운로드하지 않는다.
python -m A2.inference --eval-checkpoint A2/runs/pretrained/best.pt --data-root data --output A2/runs/eval

# A3 paired 평가
python -m A3.inference --checkpoint A2/runs/pretrained/best.pt --output A3/runs/eval

python -m pytest tests A3/tests A4/tests B2/tests -q
```

모든 명령은 저장소 루트에서 실행한다. 기존 `train --eval-checkpoint`도 해당 실험의
`inference.main`으로 전달해 호환하지만 새 코드는 `inference` 진입점을 사용한다.
기존 checkpoint의 parameter 키와 실험 이름은 유지한다. 새 QAT state에는 quantizer별
`_bit_width` buffer가 추가된다. `pretrained` 키가 없는 checkpoint는
scratch로 해석하며, 재개 시에는 원래의 학습 옵션과 `--pretrained` 설정을 일치시킨다.

## 실험별 재현성 검증 범위

```powershell
python -m pytest tests/test_experiment_reproducibility.py -q
```

이 테스트는 모델 factory, optimizer, 학습·추론 함수를 실제 구현으로 실행한다.
데이터 공급만 고정 합성 이미지 3장으로 대체하며 CPU, workers=0에서 검증한다.

- A1/A2/A4/B1/B2: 64×64 입력, 2 epoch, 같은 seed의 독립 실행을 비교한다.
  DropPath=0.1, Mixup=0.8, batch size=2, gradient accumulation=2로 난수 사용과 마지막 홀수 batch를 포함한다.
  QAT는 epoch 1부터 observer를 동결하고, pruning 실험은 pruning rate=1을 사용한다.
- 각 학습 실험은 epoch 0 checkpoint 저장 후 의도적으로 중단하고 `last.pt`에서 재개한다.
  연속 학습·반복 학습·재개 학습의 `last.pt`와 `best.pt`에서 모델/observer buffer,
  optimizer, RNG, epoch, 선택 점수, split hash를 정확히 비교한다. epoch 로그도 같아야 한다.
  독립 inference 진입점에서 best checkpoint를 복원하여 loss, 정확도, token 수를 비교한다.
- A3: 실제 A2 W2A8 모델을 합성 입력으로 보정하여 저장한 checkpoint로 224×224 paired 추론을 두 번 실행한다.
  dense/pruned 지표와 token 경로가 일치하고 checkpoint 파일 및 평가 중 모델 상태가 변하지 않아야 한다.
  A3는 추론 전용이므로 학습·재개 대상이 아니다.

시간 측정값은 정확 일치 비교에서 제외한다. 이 검증은 Food-101 전체 데이터·기본 300 epoch의
최종 정확도 재현, 리팩토링 전 코드와의 전체 수치 동등성, GPU/다중 worker 재현성,
TensorRT 배포 성능을 입증하지 않는다. 실제 pretrained 다운로드/복원 검증도 아래 별도 테스트로 구분한다.

### 리팩토링 전후 직접 비교

```powershell
python -m pytest tests/test_refactor_equivalence.py -q --tb=short
```

기준은 리팩토링 직전 커밋 `c52ca444263e88b26ccc173cc449224edb986cc6`이다.
테스트가 `git archive`로 이전 `vitqat`, `A3`, `cropr` 소스를 임시 디렉터리에 그대로 추출하고,
이전/현재 코드를 각각 독립 Python 프로세스에서 실행한다. 따라서 해당 커밋을 포함하는 Git 이력이 필요하다.
모델·학습 함수·추론 함수는 대체하지 않고, 고정 합성 TensorDataset만 공급한다.
이전 optimizer 구성은 이전 runner의 parameter grouping과 AdamW 설정을 사용한다.

검증 조건은 CPU, 실제 ViT-Tiny, 입력 3장, 2 epoch, batch=2, accumulation=2,
DropPath=0.1, Mixup=0.8이다. 학습 QAT는 W2A2이며 두 번째 epoch부터 observer를 동결한다.
A4/B1/B2 pruning rate는 1이다. 시간 측정이나 출력 파일 형식이 아닌, 초기 가중치,
epoch별 loss·정확도·logits, 최종 모델/observer·optimizer·RNG 상태를 정확 일치로 비교한다.
추가된 `_bit_width` 메타데이터만 이전 모델 state 비교에서 제외한다.
A3는 이전 A2 W2A8 checkpoint를 현재 A3에서 복원하여 224×224 dense/pruned logits와 token 경로를 비교한다.

2026-11-15 실행 결과: **8 passed, 3 failed** (198.55초).

| 비교 조건 | A1 | A2 | A3 | A4 | B1 | B2 |
|---|---|---|---|---|---|---|
| 같은 seed로 생성 후 학습 | 실패 | 실패 | 추론 전용 | 실패 | 정확 일치 | 정확 일치 |
| 학습 직전 RNG까지 동일하게 설정 | 정확 일치 | 정확 일치 | 추론 전용 | 정확 일치 | 정확 일치 | 정확 일치 |
| 동일 checkpoint의 dense/pruned 추론 | — | — | 정확 일치 | — | — | — |

A1/A2/A4의 초기 가중치는 같지만 모델 생성 직후 Torch RNG 상태가 다르다.
기존 코드는 dense 모델에서도 Cropr adapter 모델을 추가 생성한 뒤 backbone 가중치를 복사했다.
현재 코드는 이 불필요한 생성을 제거하여 난수 소비량이 줄었고, 이후 DropPath 난수와 학습 결과가 달라진다.
예를 들어 A1 첫 epoch loss는 이전 `4.636846542358398`, 현재 `4.5360541343688965`였다.
학습 직전 RNG를 동일하게 맞추면 다섯 학습 실험 모두 loss/logits/모델/optimizer/RNG가 정확히 일치한다.

따라서 **같은 seed만으로 리팩토링 전후 학습 궤적이 보존된다는 검증은 통과하지 못했다.**
이 차이를 숨기지 않도록 해당 세 테스트는 skip/xfail 처리하지 않고 실패로 유지한다.
이 테스트는 모델 구성·학습 step·평가 연산 비교이며, 이전 CLI/runner 전체 실행이나
Food-101 전체 학습의 성능 동등성을 보장하지 않는다.

### 전체 의존성 설치 후 검증

2026-11-15 `vitqat` 가상환경에 `requirements.txt`의 누락 패키지
`tensorrt-cu12==10.13.3.9`, `onnx==1.17.0`, `onnxruntime==1.20.1` 및 전이 의존성을 설치했다.
`pip check`가 통과했고, Windows / PyTorch 2.3.0+cu121 / RTX 3060 환경에서
TensorRT import와 GPU builder 생성도 확인했다.

```powershell
python -m pip install -r requirements.txt
python -m pip check
$env:RUN_PRETRAINED_INTEGRATION = "1"
python -m pytest tests A3/tests A4/tests B2/tests -q -ra --tb=short --junitxml=runs/validation/full-dependencies.xml
Remove-Item Env:RUN_PRETRAINED_INTEGRATION
```

전체 실행은 230개 중 **224 passed, 6 failed, 0 skipped**였다(733.63초).
이 중 B2 checkpoint 파일 쓰기 오류 3건은 코드 변경 없이 해당 테스트만 재실행하여
**3 passed**로 확인했다(73.64초). 재실행 보고서는
`runs/validation/checkpoint-write-retry.xml`이다. 일시적 파일 쓰기 오류의 원인은 확정하지 않았다.

재검증까지 반영한 상태는 **227개 통과, 3개 실패, 건너뜀 0개**다.
남은 세 실패는 위에서 설명한 A1/A2/A4의 같은-seed 전후 비교이며 패키지 누락 문제가 아니다.
기존에 의존성 부족으로 건너뛰었던 9개 ONNX/ONNX Runtime/TensorRT 테스트는 모두 실행되어 통과했다.
실제 pretrained 가중치 테스트 6개도 통과했다. TensorRT 검증에는 FP32·양자화 레이어 및
A3 dense/pruned 모델의 실제 엔진 생성·GPU 실행이 포함된다.
이는 합성 입력 기반 기능 검증이며 Food-101 전체 학습 성능 재현 결과는 아니다.

## 모델 복원 계약

각 실험의 `model.py`가 `build_model(args)`와 `restore_model(model, checkpoint)`를 소유한다.
일반 추론, 학습 재개, best checkpoint 재평가, TensorRT 변환 준비가 같은 로컬 복원 함수를
사용한다. utils는 실험의 복원 함수를 호출하며 FP32/QAT 선택이나 quantizer 타입을 알지 못한다.
TensorRT 변환은 QAT 복원 이후 해당 실험의 export 코드에서 수행한다.

QAT 복원은 mode, Cropr/CLS 정책, weight/activation bit, QAT scope를 검증한다.
state의 비트 수가 config와 다르거나 observer가 미보정·비정상 상태이거나 fake quantization이
비활성화되어 있으면 평가 전에 실패한다. QAT를 이미 적용한 모델에 다시 적용하는 것도 거부한다.
새 `_bit_width` buffer가 없는 기존 checkpoint는 원본 config에 저장된 비트 수로 복원한다.
구형 state만으로 비트 수를 알아낼 수는 없으므로 config 없는 QAT checkpoint를 추측해서 읽지 않는다.

## 확장 위치

- pretrained 가중치·새 backbone·head-only 등은 해당 실험 `model.py`에서 결정한다.
- 새로운 저장 형식이나 양자화 표현은 해당 실험의 `restore_model`에서 복원한다. 공통 loader에 실험 분기를 추가하지 않는다.
- optimizer 또는 loss 변경은 해당 실험 `train.py`에서 결정한다. 학습 루프·criterion·Mixup·scheduler·gradient clipping도 해당 실험의 `train_epoch`에 있다.
- quantizer, bit 정책, observer 동작은 해당 실험 `quantization.py`와 `train.py`에서 변경한다.
- 추론 전용 변환이나 backend는 해당 실험 `inference.py` 또는 `tensorrt_int8.py`에서 변경한다.

pretrained 가중치는 timm의 호환 ViT에 대해 지원하며 Food-101 전처리를 사용한다.
B1/B2는 원본 Cropr adapter 구조(단일 CLS, MLP ratio 4 등)의 제한을 유지한다.
CLS pruning과 QAT도 각 구현의 ViT 연산 규약을 요구하므로 임의 모델 지원을 뜻하지 않는다.
기본 테스트는 공급한 가중치의 보존, 로컬 factory 교체, resume/평가 시 재요청 방지,
A2→A3 state dict 호환을 확인한다. Hub API 호환성도 다운로드 없이 검사한다.

실제 pretrained 가중치를 사용하는 통합 테스트는 별도로 실행한다. 모델 생성 함수를 대체하지 않으며,
처음 실행할 때 timm/Hugging Face cache로 가중치를 내려받을 수 있다.

```powershell
$env:RUN_PRETRAINED_INTEGRATION = "1"
python -m pytest tests/test_real_pretrained.py -q -s
Remove-Item Env:RUN_PRETRAINED_INTEGRATION
```

`vit_tiny_patch16_224.augreg_in21k_ft_in1k`로 6개 테스트를 통과했다.
A1/A2/A4/B1/B2의 backbone 파라미터 150개가 원본 pretrained 가중치와 정확히 일치하며,
Food-101 head의 `(1, 101)` forward 출력이 유한한 값을 갖는 것을 확인했다.
A3는 실제 pretrained 기반 A2 checkpoint를 복원한 뒤 logits가 정확히 일치함을 확인했다.
QAT observer는 합성 입력으로 초기화한 것이며, Food-101 전체 학습이나 정확도를 검증한 결과는 아니다.

## CLI와 의존성

추론 CLI에는 `--epochs`, `--lr`, `--mixup`, `--resume`, `--model` 등 학습·모델 설정 인자가 없다.
모델·정밀도·전처리 설정은 checkpoint에서 복원한다. 각 실험 `config.py`의
`CHECKPOINT_REQUIRED`는 필수 구조·양자화 설정이고, `CHECKPOINT_DEFAULTS`는 구형 checkpoint의
비구조적 호환 기본값만 제공한다. mode나 QAT bit가 없으면 기본값으로 채우지 않는다.

모든 Python 의존성은 루트 `requirements.txt` 한 파일에서 관리한다.
`environment.yml`은 Python 3.10 환경을 만들고 같은 requirements를 설치하는 Conda 진입점이다.
실제 로딩에서 timm 1.0.3과 huggingface_hub 2.2.0의 API 불일치가 재현되어,
호환성과 실제 로드를 확인한 `huggingface_hub==0.36.2`를 명시적으로 고정했다.

```powershell
python -m pip install -r requirements.txt
```
