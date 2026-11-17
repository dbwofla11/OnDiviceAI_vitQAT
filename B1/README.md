# B1: Cropr + Linear QAT (LSQ/LSQ+)

원본 Cropr ViT(`cropr/cls`)에 **Linear layer QAT**를 붙여 QAT와 learned-query token pruning을 동시에 학습한다.
기본 설정은 **INT2 (W2A2)** 이다. `cropr/`의 원본 파일은 수정하지 않는다.
A1과 비교하면 pruning과 양자화를 함께 적용한 영향을, B2와 비교하면 같은 Cropr 구조에서 양자화 방식(LSQ Linear QAT 대 affine full QAT)의 영향을 확인할 수 있다.

현재 데이터셋은 Food-101(101 classes), 기본 모델은 `vit_tiny_patch16_224`, 입력 크기는 224다.
같은 seed에서 B2·A1과 데이터 split, 학습 recipe, 초기 FP32 weight가 동일하다(테스트로 확인).

**학습과 추론은 같은 quantizer 모듈을 사용한다.**
- weight는 대칭이다(signed, zero point 0).
- activation은 기본 LSQ+(비대칭, 학습 가능한 offset)이며, `--activation-quantizer lsq`를 주면 대칭이다.
- 어느 쪽을 고르든 학습과 추론에 같은 방식이 적용된다. 추론용으로 양자화 grid를 바꾸는 변환은 없다.

## 폴더 구조

```text
B1/
  __init__.py       # Python package
  config.py         # CLI, 실험별 기본값, checkpoint 설정 검증
  model.py          # 모델 생성, scratch/--pretrained/--init-checkpoint 초기화, Linear QAT 적용, checkpoint 복원
  quantizers.py     # LSQ / LSQ+ / StatsQ fake quantizer
  modules.py        # QuantLinear, QuantConv2d, QuantCrossAttention(Cropr scorer), prepare_linear_qat(), 복원 검증
  train.py          # optimizer, loss(main + aux CE), 학습 루프, LR·증강·best 선택 정책
  inference.py      # 저장된 checkpoint의 test split 평가
  metrics.py        # quantizer 파라미터를 제외한 params, packed storage 추정, 결과 항목·model_state.pt 저장
  cropr_adapter.py  # 이 실험의 timm → 원본 Cropr 모델 조립·가중치 전달
  tests/            # 수식·참조 구현 대조, 모델 변환, 파이프라인 검증
  README.md         # 이 실험의 실행 안내
```

`utils`는 데이터 split, 학습/평가 runner, checkpoint/resume, 결과 파일 같은 공통 기능에만 사용한다. 모델·학습·추론·양자화 정책은 이 폴더가 소유한다.

## 1. 기존 방법론 조사

| 방법 | Quantizer / scale | ViT에서 양자화한 범위 | 첫/마지막 layer | 초기화·학습 |
|---|---|---|---|---|
| 저장소 A2/A4/B2 | PyTorch `FakeQuantize`(비대칭 affine), min/max observer(EMA), scale 비학습 | 모든 Linear/Conv + attention Q/K/V/prob (+ B2는 Cropr) | 동일 bit | scratch 또는 pretrained |
| [LSQ](https://arxiv.org/abs/1902.08153) (ICLR'20) | 학습 가능한 step size `s`, `g = 1/sqrt(N·Qp)` gradient scale | 모든 conv/FC의 weight·입력 | **8-bit 유지** | FP 모델에서 시작. `s = 2<|v|>/sqrt(Qp)` |
| [LSQ+](https://arxiv.org/abs/2004.09576) (CVPRW'20) | LSQ + **학습 가능한 offset β**(비대칭 activation) | 동일 | 동일 | weight `s = max(|μ±3σ|)/2^(b-1)`, activation은 MSE 초기화. W2A2에서 이득 최대 |
| [Q-ViT](https://arxiv.org/abs/2210.06707) (NeurIPS'22) | LSQ 계열, activation은 zero point를 둔 비대칭 | Linear + attention 행렬곱 | **patch embed/head 8-bit** | FP 모델 + distillation, 300 epochs |
| [OFQ](https://arxiv.org/abs/2302.02210) (ICML'23) | "LSQ가 사실상 표준". 2-bit용 **StatsQ**(`2·mean|W|` 통계 scale) + CGA, QKR | Linear + attention 행렬곱 | **patch embed/head 8-bit** | FP 모델 + KD. DeiT-S W2A2: LSQ 68.0 → StatsQ 74.3 |
| PyTorch `_LearnableFakeQuantize` | LSQ kernel, `grad_factor = 1/sqrt(numel·qmax)` | – | – | – |

OFQ와 Q-ViT의 공식 코드(`nbasyl/OFQ`, `YanjingLi0202/Q-ViT`)도 확인했다.
- 두 구현 모두 논문 pseudocode의 `round_pass`/`grad_scale` detach trick을 사용한다.
- activation gradient scale은 `x.numel()`(batch 포함)을 쓴다.
- OFQ의 per-channel weight는 `N = in_features`를 쓴다.

**결론.**
- QAT 방법은 LSQ 계열을 기준선으로 쓴다.
- INT2에서는 다음 세 가지가 성능을 크게 좌우한다.
  - 비대칭 activation(offset)
  - 초기화
  - 첫/마지막 layer 8-bit
- 문헌의 저비트 QAT는 모두 학습된 FP 모델에서 시작한다.

## 2. 구현 설계

**양자화 대상 (Linear QAT).**
- Transformer block의 모든 `nn.Linear`(qkv, attn proj, fc1, fc2)는 `W{weight-bits}A{activation-bits}`로 양자화한다.
- patch embedding(16×16 patch에 대한 linear)과 classifier head는 `--first-last-bits`(기본 8)로 둔다.
- Cropr scorer는 `--cropr-scorer-bits`(기본 8)로 둔다.
  - scorer는 `scores = Σ X·Qᵀ`인 linear map이다. learnable query `Q`를 weight로, token `X`를 입력으로 양자화한다.
  - 학습 경로(`forward`)와 추론 경로(`forward_scorer`)가 같은 quantizer를 공유한다. 따라서 학습과 배포에서 같은 양자화 점수로 token을 고른다.
  - 224px ViT-Tiny에서 전체 연산의 0.035%뿐이지만 순위가 민감한 연산이다. 그래서 첫/마지막 layer와 같은 이유로 8-bit를 기본값으로 한다. 2-bit에서는 동점 점수도 많아진다. 전부 INT2로 하려면 `--first-last-bits 2 --cropr-scorer-bits 2`를 쓴다.

**FP32 유지.**
- attention QKᵀ/AV 행렬곱, softmax, LayerNorm, GELU, residual, CLS/pos embedding
- 추론 때 제거되는 Cropr aggregator/보조 head

**Quantizer.**
- Weight는 출력 채널별 signed `[-2^(b-1), 2^(b-1)-1]`로 양자화한다(대칭, zero point 0).
  - 기본값 `lsq+`: LSQ step에 LSQ+ 3σ 초기화를 쓴다.
  - `lsq`: LSQ 원 논문 초기화를 쓴다.
  - `statsq`: OFQ 방식이다(대칭 half-integer level). block 내부 저비트 Linear에만 적용하고, 8-bit layer는 `lsq+`를 쓴다.
- Activation은 tensor별 static scale로 양자화한다. integer GEMM에서 scale을 matmul 밖으로 뺄 수 있는 형태다.
  - 기본값 `lsq+`: `x̂ = clamp(round((x-β)/s), 0, 2^b-1)·s + β`(비대칭). β는 inference 때 bias로 접힌다(LSQ+ Eq. 5).
  - `lsq`: signed `[-2^(b-1), 2^(b-1)-1]`(대칭), `s = 2<|x|>/sqrt(Qp)`.
  - 초기화: 첫 학습 batch에서 `||x̂ - x||²`를 최소화하는 clipping 범위를 고른다. 후보 범위에 LSQ+ min/max 초기화(Eq. 7)가 포함된다.
- 학습과 추론은 같은 quantizer 모듈을 쓰므로 위 방식이 그대로 추론에도 적용된다.
- Gradient는 다음을 따른다.
  - LSQ Eq. 3/5와 LSQ+ Eq. 3/4
  - gradient scale `g = 1/sqrt(N·Qp)`, offset에도 같은 `g`
  - step size가 0 이하로 내려가지 않게 하는 OFQ식 clip
- step/offset 파라미터는 1-D다. 그래서 `utils.optim.weight_decay_groups`에서 weight decay 대상이 아니다.
- weight step은 변환 시점의 weight로 초기화한다(`--pretrained` 또는 `--init-checkpoint`를 로드한 뒤).
- activation step은 첫 학습 batch에서 초기화하고, `initialized` buffer로 checkpoint에 저장한다.

**학습 loss.** 원본 Cropr와 같다: `main CE + aux_loss_weight·Σ aux CE`. 모든 forward가 fake quantization을 거친다. 따라서 `θ`(weight, step, offset)와 pruner `θ_p`(queries)가 양자화된 그래프에서 함께 최적화된다.

## 3. 실행

아래 명령은 모두 **저장소 루트**에서 실행한다. `cd B1` 후 직접 `train.py`를 실행하지 않는다.
환경과 데이터는 [docs/experiments.md](../docs/experiments.md)와 같다. `data/food-101/{images,meta}`를 읽는다.

```powershell
conda env create -f environment.yml
conda activate vitqat

# 첫 실행에서 --download로 Food-101을 data/에 받는다(약 5GB).
# Cropr(rate 8) + Linear QAT W2A2, 첫/마지막·scorer 8-bit, scratch:
python -m B1.train --data-root data --download --output B1/runs/w2a2 --device cuda

# 권장(문헌 방식): 학습된 FP32 모델에서 QAT fine-tuning
# (a) timm pretrained backbone에서 시작
python -m B1.train --pretrained --model vit_tiny_patch16_224.augreg_in21k_ft_in1k --data-root data --output B1/runs/w2a2_pretrained --device cuda
# (b) A1에서 학습한 FP32 checkpoint로 초기화(Cropr query·보조 head는 새로 초기화)
python -m A1.train --data-root data --output A1/runs/fp32 --device cuda
python -m B1.train --init-checkpoint A1/runs/fp32/best.pt --epochs 100 --warmup-epochs 5 --lr 1e-4 --data-root data --output B1/runs/w2a2_ft --device cuda

# 비교/ablation 예
python -m B1.train --no-use-cropr --data-root data --output B1/runs/lqat_w2a2 --device cuda    # Linear QAT only
python -m B1.train --weight-quantizer statsq --data-root data --output B1/runs/w2a2_statsq --device cuda
python -m B1.train --weight-bits 4 --activation-bits 8 --data-root data --output B1/runs/w4a8 --device cuda
```

- 새 학습의 `--output`은 없거나 비어 있는 디렉터리여야 한다.
- 학습은 매 epoch validation을 수행하고, Top-1이 가장 높은 checkpoint를 `best.pt`로 선택해 test split을 평가한다.
- `--init-checkpoint`는 같은 구조의 FP32 state를 받는다(`best.pt`/`last.pt`/`model_state.pt`).
  - A1(dense)이나 이전 B1(FP32 Cropr) checkpoint를 쓸 수 있다.
  - 양자화 상태가 든 checkpoint는 거부하고, `--pretrained`와 함께 쓸 수 없다.

### 중단된 학습 재개

```powershell
python -m B1.train --data-root data --output B1/runs/w2a2 --resume B1/runs/w2a2/last.pt --device cuda
```

같은 output의 `last.pt`와 `best.pt`를 함께 보존한다. bit·quantizer·Cropr·증강 등 원래 학습 옵션을 그대로 전달한다.
재개와 추론에서는 저장된 state를 복원하므로 pretrained 가중치나 `--init-checkpoint` 파일을 다시 읽지 않는다.

## 추론·평가

```powershell
python -m B1.inference --eval-checkpoint B1/runs/w2a2/best.pt --data-root data --output B1/runs/eval --device cuda --batch-size 128 --workers 4 --latency-warmup 10 --latency-repeats 100
```

- Food-101 test split 전체를 평가한다.
- bit·quantizer 종류·Cropr 설정·전처리는 checkpoint config에서 복원하므로 다시 넘기지 않는다.
- 복원 시 activation quantizer가 초기화되지 않았거나 state에 비유한 값이 있으면 평가 전에 거부한다.
- 이전 `cropr_linear_qat` 패키지로 저장한 checkpoint는 config에 `mode`가 없어 거부한다.

## 주요 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--weight-bits`, `--activation-bits` | 2, 2 | block 내부 Linear의 weight/입력 bit (2/3/4/8) |
| `--first-last-bits` | 8 | patch embedding, head (32 = FP32) |
| `--cropr-scorer-bits` | 8 | Cropr scorer query/token (32 = FP32) |
| `--weight-quantizer` | `lsq+` | `lsq+` / `lsq` / `statsq` (모두 대칭) |
| `--activation-quantizer` | `lsq+` | `lsq+`(비대칭+offset) / `lsq`(대칭 signed) |
| `--use-cropr`, `--cropr-pruning-rate`, `--cropr-llf`, `--aux-loss-weight` | on, 8, off, 1.0 | 원본 Cropr 설정(B2와 동일 의미). `--no-use-cropr`는 Linear QAT-only ablation |
| `--init-checkpoint` | 없음 | 새 학습을 FP32 checkpoint로 초기화. 양자화 상태가 든 checkpoint는 거부 |
| `--pretrained` | off | timm pretrained backbone (예: `--model vit_tiny_patch16_224.augreg_in21k_ft_in1k`) |
| `--model`, `--image-size` | `vit_tiny_patch16_224`, `224` | 학습 모델과 입력 크기 |
| `--epochs`, `--warmup-epochs` | `300`, `20` | 학습 기간과 LR warmup |
| `--lr`, `--min-lr`, `--weight-decay` | `5e-4`, `1e-6`, `0.05` | AdamW와 cosine LR 설정 |
| `--batch-size`, `--accum-steps` | `128`, `1` | batch 크기와 gradient accumulation 횟수 |
| `--seed`, `--val-fraction` | `42`, `0.1` | 난수 seed와 validation 비율 |
| `--mixup`, `--cutmix`, `--random-erasing`, `--crop-min-scale` | `0`, `0`, `0`, `0.08` | 추가 증강. 기본은 꺼짐 |
| `--workers`, `--threads` | `4`, `0` | DataLoader worker와 CPU thread 수 |
| `--latency-warmup`, `--latency-repeats` | `10`, `100` | batch=1 forward 지연시간 측정 횟수 |

공통 학습 옵션과 기본값은 B2와 같다(테스트로 고정). 기본 recipe는 300 epochs, AdamW, LR 5e-4, cosine, batch 128이다.

## 출력 파일

| 파일 | 내용 |
|---|---|
| `config.json`, `split.json` | 실행 설정과 데이터 split 정보 |
| `epochs.jsonl` | epoch별 total/main/aux train loss, LR, validation 지표(학습 시 생성) |
| `last.pt` | 마지막 epoch의 모델·quantizer·optimizer·RNG·설정, 재개용 |
| `best.pt` | 최고 validation Top-1 checkpoint, 평가용 |
| `metrics.json`, `metrics.csv` | test 정확도·loss·모델 크기·연산량·지연시간 |
| `model_state.pt` | 선택된 모델의 state dict(FP32 master weight와 quantizer 상태) |

`metrics.json`의 형식은 다른 실험과 같고, 다음 필드가 추가된다.

| 필드 | 의미 |
|---|---|
| `mode` | `linear_qat` |
| `weight_bits`, `activation_bits`, `first_last_bits`, `cropr_scorer_bits` | 각 그룹의 bit 수 |
| `weight_quantizer`, `activation_quantizer` | 사용한 quantizer |
| `quantized_layers` | 양자화된 layer 목록 |
| `quantizer_params` | step/offset 개수. `params`, `inference_params`에서 제외 |
| `inference_storage_estimated_packed_bytes` | 추론 때 제거되는 Cropr aggregator/보조 head를 뺀 packed storage 추정 |
| `initialization` | `scratch` / `timm_pretrained` / `init_checkpoint` |

## 4. 검증

```powershell
python -m pytest B1/tests tests/test_quantization_consistency.py -q
```

- **수식 대조**
  - LSQ weight/activation의 출력과 `x`·`s` gradient를 논문 Eq. 1–3, 5를 그대로 쓴 참조식과 비교한다.
  - 같은 값을 PyTorch 내장 LSQ kernel(`torch._fake_quantize_learnable_per_{channel,tensor}_affine`)과도 비교한다. 2/3/4/8-bit 모두 일치한다.
  - 두 정의는 clip 경계 바깥 반 step 구간에서만 다르다. LSQ는 clip 후 round하고, PyTorch kernel은 round한 값으로 mask한다. 테스트는 이 구간을 제외하고 비교한다.
  - LSQ+의 `s`, `β` gradient(Eq. 3/4)를 확인한다.
  - min/max 초기화(Eq. 7)를 확인하고, MSE 초기화가 2-bit에서 오차를 줄이는지 확인한다.
  - StatsQ 결과가 OFQ 공식 코드와 일치하는지 확인한다.
- **모델 변환**
  - 양자화 범위와 bit 배정을 확인한다.
  - 같은 seed에서 B2의 FP32 Cropr backbone, A1 dense 모델과 FP32 파라미터가 같은지 확인한다.
  - quantizer를 끄면 원본 출력과 일치하는지 확인한다(train의 보조 출력 포함, LLF on/off).
  - scorer의 학습/추론 점수가 같은지, 모델 전체의 학습/추론 logits가 같은지 확인한다(`lsq+`, `lsq`).
  - W2A2/W4A8 학습 step에서 weight, step, offset, query가 gradient로 갱신되는지 확인한다.
  - init-checkpoint(B2 backbone, A1 dense)와 pretrained 로딩을 확인한다. 재개·추론에서는 init 파일을 읽지 않는다.
  - 미초기화·비유한 state, 다른 실험의 checkpoint, 중복 변환을 거부하는지 확인한다.
- **지표**
  - params, inference_params, GFLOPs, token 수가 B2 규약으로 잰 FP32 Cropr 모델과 같은지 확인한다.
  - W4−W2 storage 차이가 block weight 수 × 2bit와 정확히 같은지 확인한다.
- **파이프라인**
  - 실제 torchvision Food101 loader를 쓰는 생성 이미지 fixture로 학습→검증→저장→test→resume→`B1.inference` 재평가를 확인한다.
  - checkpoint 재개 후 학습이 bit 단위로 같은지 확인한다.
  - 잘못된 옵션은 출력 폴더를 만들기 전에 거부되는지 확인한다.
- 저장소 공통 테스트(`tests/test_experiment_entrypoints.py`, `tests/test_experiment_reproducibility.py` 등)도 B1의 학습·재개·추론 진입점을 실행한다.

**이전 결과와의 동등성 (2026-11-17).**
- 비교 대상은 이전 `cropr_linear_qat` 원본 코드다. 리팩토링 전 저장소 커밋 `bf76a21`의 `vitqat` 위에서 실행했다.
- 두 코드를 각각 별도 프로세스에서 같은 입력·seed로 2 epoch 학습했다(64px, batch 2, accumulation 2, DropPath 0.1).
- 다음 6개 옵션 조합을 비교했다: 기본 W2A2, `--cropr-llf`, W4A8+`statsq`+`lsq`, W3A4+`lsq`, 첫/마지막·scorer FP32, `--no-use-cropr`.
- 모든 조합에서 다음 항목이 **정확히 일치**했다: 초기 가중치·RNG, epoch별 loss·정확도·logits, 최종 모델·optimizer·RNG 상태, 지표.
- 환경은 Conda `vitqat`(Python 3.10.22, PyTorch 2.3.0+cu121, torchvision 0.18.0, timm 1.0.3), CPU다.
- `python -m pytest B1/tests -q` → **41 passed, 1 skipped**(CUDA).
- 원래 `cropr_linear_qat` 테스트 38개(CUDA 1개 포함)는 import 경로만 바꾸거나 저장소 구조에 맞춰 이식했다.
- 학습/추론 일치, A1 초기값, 복원 검증 테스트 4개를 추가했다.

**이전 `cropr_linear_qat`에서의 결과 (2026-10-23).** 이 머신에는 NVIDIA GPU가 없어 CPU에서 검증했다.
- 추가 확인 (W2A2, 224px, 기본 설정).
  - **GFLOPs와 pruning**: GFLOPs 1.893, token pruning 0.454로 FP32 Cropr와 같다. bit 수는 GFLOPs를 바꾸지 않는다.
  - **추론 경로 packed storage**: 1.86 MB다(FP32 22.2 MB, W4A8 3.18 MB). 보조 분기를 포함하면 15.8 MB다.
  - **LSQ update 비율**: `R = (|∇s|/s)/(‖∇w‖/‖w‖)`는 block weight step에서 0.2–0.35다. gradient scale이 없으면 3–10이다. 학습은 AdamW라 gradient 크기가 정규화되므로 `g`의 영향은 SGD보다 작다.
    - activation step gradient의 중앙값은 6e-6으로 Adam eps(1e-8)보다 충분히 크다.
- 합성 10-class 데이터(64px, seed 1개)로 학습 가능성을 확인했다.
  - FP32, W4A8, W2A2(`lsq+`, `lsq`, `statsq`) 모두 train loss가 4.0에서 0.001로 줄었다.
  - test top-1은 95–98.5%였다.
  - 장난감 데이터라 quantizer 간 우열의 근거는 아니다. Food-101에서 ablation으로 비교한다.
- 실제 Food-101로 짧게 실행했다(2026-10-24, CPU, 기본 W2A2, 224px).
  - torchvision `Food101(download=True)`로 MD5를 확인한 데이터를 썼다.
  - split은 문서와 같은 68,175 / 7,575 / 25,250이다.
  - 클래스별 train 2장, val/test 1장 subset으로 1 epoch(7 step)를 돌렸다.
  - 학습→검증→test→지표 저장이 끝까지 동작했다. pruning ratio는 0.454, 추론 경로 storage는 1.86 MB였다. 정확도는 의미가 없는 길이다.

## 5. 한계와 주의

- **Fake quantization이다.** 실제 INT2 packed kernel이나 export가 아니다. latency는 `fake_quant_fp32`이고, storage는 이론 추정치다.
- **INT2 scratch QAT는 어렵다.**
  - 문헌(LSQ, LSQ+, Q-ViT, OFQ)은 모두 학습된 FP 모델에서 시작하고, ViT에서는 KD도 쓴다.
  - 기본값은 저장소 프로토콜(scratch, 300 epochs)을 따른다.
  - 정확도가 낮으면 `--pretrained` 또는 `--init-checkpoint`(A1 FP32 → QAT)를 먼저 시도한다.
  - LSQ는 저비트일수록 낮은 weight decay가 유리하다고 보고했다(2-bit에서 1/4).
- **미구현**
  - KD
  - OFQ의 CGA(진동 weight 고정)와 QKR
  - attention 행렬곱 양자화(Linear QAT 범위 밖)
- 이 머신에는 NVIDIA GPU가 없다. 그래서 CUDA 테스트는 skip되었고, 실제 전체 학습과 RTX 4090 latency는 검증하지 않았다.

## 도움말

```powershell
python -m B1.train --help
python -m B1.inference --help
```

[전체 실험 구조와 공통 코드 경계](../docs/experiment-folder-structure.md), [학습 조건·지표 정의](../docs/experiments.md).
