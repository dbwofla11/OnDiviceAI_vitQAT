> **Current implementation (2026-10-27):** The following is the previous learned-query design and paper notes. A3 inference now selects tokens using CLS self-attention Top-K, retaining the Cropr removal schedule and LLF. No query checkpoint is required. See [execution](execution.md).

# 구현 아키텍처: 추론 Cropr router

상태: **추론 평가 코드 구현**. 실제 query 확보와 task 성능 평가는 남아 있다. 논문 기준은 [pruning 파라미터](pruning-parameters.md), 비교 조건은 [실험 설계](experiment-design.md), CLI와 query 계약은 [실행 방법](execution.md)에서 관리한다.

## 1. 입력과 확장 경계

실험 전용 평가 코드는 `A3/`, 문서는 `docs/A3/`, 전용 테스트는 `A3/tests/`에 둔다. A3 전용 pruning과 지표는 `A3/pruning/`, `A3/metrics.py`로 분리했다. `cropr/` 원본은 유지한다.

모델·데이터·pooling·QAT bit/구현·학습 및 평가 설정은 앞선 QAT-only 실험의 실제 config/checkpoint를 그대로 따른다. 이 문서의 Small/16·224·W4A8은 예상 예시이며 별도 고정 요구사항이 아니다. 구현은 상속한 모델 구조와 추가 pruning 설정의 호환성을 검사하며, 예시와 다르다는 이유로 backbone이나 학습 설정을 변경하지 않는다.

이 작업은 다른 담당자의 QAT-only checkpoint와 별도 학습된 Cropr queries를 입력으로 받는다. Backbone 학습과 scorer 학습은 이 평가 경로에 넣지 않는다. Query 확보 방식이 결정되지 않았으므로 가짜 query 초기화로 빈 부분을 메우지 않는다.

```text
외부 선행 작업: QAT-only best checkpoint + config/split/metrics
별도 필수 입력: 같은 backbone에 맞는 학습된 Cropr queries + provenance

신규 paired 평가
  원래 설정 복원 → dense QAT 모델 생성 → strict backbone state load
  → observer 정지, eval → dense 평가
  → 학습된 router 연결 → pruned forward + LLF → 비교 결과 저장
```

Dense/pruned의 backbone weight와 quantizer buffer는 동일하며 평가 전후 불변이어야 한다. Router query는 별도 state로 관리한다. 기존 QAT 모델의 state dict에 임의 missing key를 허용하지 않는다.

## 2. 모듈 책임

공개 기능은 `A3/pruning/`으로 분리했다. `inference.py`는 기존 추론 전용 기능, `model.py`는 `attach_pruning`과 학습/추론 모드 wrapper, `__init__.py`는 공개 export를 제공한다. `A3/inference_pruning.py`는 호환 import만 유지한다. 평가 CLI는 공개 API를 호출하고, 학습 모드의 `PrunedOutput`/보조 loss 계약은 [라이브러리 API](../library-api.md)를 따른다. A3 기본 평가는 여전히 추론 pruning이며, 신규 학습 모드는 재사용 기능이다.

| 파일 | 현재/예정 | 책임 |
|---|---|---|
| `A3/model.py` | 기존 재사용 | 전달 config로 dense QAT 모델 생성 |
| `A3/quantization.py` | 기존 유지 | QAT backbone 연산, CLS attention callback 추가 불필요 |
| `utils/checkpoint.py`, `utils/classification.py` | 재사용 | 원래 config와 split 기준 strict checkpoint load, CE/Top-1 평가 |
| `A3/pruning/inference.py` | 구현 | 학습된 query router, selector, 제거 feature cache, LLF, trace |
| `A3/pruning/model.py` | 구현 | 공개 attach API, train/eval 모드 분기, 보조 출력 및 query export |
| `A3/inference_pruning.py` | 호환 | 이전 import 경로의 re-export |
| `A3/eval_pruning.py` | 구현 | 두 checkpoint 검증, paired 평가와 결과 저장 |
| `A3/metrics.py` | 확장 | router parameter/FLOPs, 실제 token trace, 전체 latency |
| `A3/tests/test_inference_pruning.py` | 추가 | scorer 원본 동등성, CLS/Top-K/LLF/trace |
| `A3/tests/test_eval_pruning.py` | 추가 | query 인수·strict load·paired 평가 검증 |

`cropr/` 원본은 보존한다. 원본 `CrossAttention.forward_scorer`, `Cropr.prune`와 LLF 동작을 재사용하거나 동등성이 검증된 wrapper로 연결한다. 기존 `runner.run()`의 학습/재개 계약을 느슨하게 만들지 않는다.

## 3. Router 상태와 forward 계약

`RouterSpec` 제안 필드: `scorer=learned-query-dot-product`, `num_queries=1`, `num_heads=1`, `embed_dim=384`, projection/LN off, `precision=fp32`.

Query 파일은 block 1~10 각각의 `[1,1,D]` tensor, schedule, 학습에 사용한 backbone SHA256, split hash와 학습 이력을 포함해야 한다. Shape, block mapping, finite 값, 출처 일치를 검증한다. 실제 제공 파일 형식이 다르면 명시적 import adapter를 만들고 누락을 random init으로 대체하지 않는다. 모든 query는 평가 중 `requires_grad=False`다.

각 pruning stage는 해당 QAT block의 출력 X로 `score = q @ X.transpose(-2,-1)`을 계산한다. CLS score를 +inf로 설정하고 `M-R_i`개를 유지한다. FP32 router는 QAT block이 출력한 FP32 fake-quant 표현을 입력으로 사용한다. Scorer 자체에 추가 quantizer를 설치하거나 backbone Q/K/probability를 읽지 않는다.

원본과 같이 `argsort(descending=True, stable=False)` 후 score 순서로 gather한다. CLS는 첫 위치, keep/removed는 분리된 feature다. 논문이 보장하지 않은 동점 규칙을 새로 추가하려면 설정과 테스트에 구현 차이로 명시한다.

## 4. Forward schedule과 LLF

예상 12-block/224/patch16 구조의 pruning config는 `after_blocks=(1,...,10)`, `remove_counts=(17,16,16,16,16,16,16,16,16,16)`, `llf=True`, `schedule_id=cropr-r16-first17-llf`다. 사용자는 1-based 번호를 쓰며 내부 인덱스 변환은 한 곳에서 수행한다. 실제 상속 모델이 다르면 해당 구조에 맞는 schedule과 query 차원을 검증하고 trace/TPR을 다시 산출한다. 지원하지 않는 구조는 명시적으로 알리고 예시 모델로 대체하지 않는다.

단계마다 제거 feature를 cache에 보관하고 keep만 다음 block으로 보낸다. 11번 block은 36 tokens를 처리한다. 이후 원본 분류 forward처럼 `[keep, removed_stage1, ..., removed_stage10]`으로 결합해 마지막 block에 197 tokens를 넣는다. Patch ID도 함께 추적해 196개 patch가 중복/누락 없이 복원됨을 확인한다. Positional embedding은 다시 더하지 않는다.

Block 입력 trace는 `[197,180,164,148,132,116,100,84,68,52,36,197]`이어야 한다. 최소 patch 수는 35, 최대 TPR 161/196, 최종 TPR 0이다. Disabled 경로는 router/scoring/cache를 완전히 건너뛰어 기존 dense 출력과 일치해야 한다. Cache와 trace는 매 forward마다 초기화한다.

## 5. 로드 및 평가 순서

1. QAT checkpoint를 `weights_only=True`로 읽고 SHA256과 원래 config를 확인한다. 구형 증강 키 보완은 기존 기본값 규칙을 유지한다.
2. 선행 config에 지정된 dense QAT 모델을 생성하고 원래 config/split을 기준으로 strict load한다. 실제 bit 수/구조, 학습 Cropr 없음 및 초기화된 observer를 확인한다. Pruning/query와의 호환성은 별도로 검사한다.
3. `eval()`과 inference mode에서 dense 평가를 수행한다. 전달된 기준 결과와 재현 여부를 기록한다.
4. Query provenance와 block별 state를 검증하고 router-only 구성을 연결한다. Aggregator/head를 추론 graph에 넣지 않는다.
5. 같은 test split에서 pruned 평가를 수행한다. Observer 재보정, optimizer 생성/step, test calibration을 수행하지 않는다.
6. Backbone/quantizer 불변과 두 checkpoint hash를 검증하고 새 평가 폴더에 저장한다.

서로 다른 backbone에서 학습된 query가 제공되면 hash 불일치를 무시하지 않는다. 해당 조합을 사용하려면 별도 실험 설계와 결과 표기가 필요하다. 현재 명령과 query 형식은 [실행 방법](execution.md)을 따른다.

## 6. 지표와 산출물

A3 전용 `measure_vit()`는 기존 지표 정의를 기준으로 `Cropr`와 A3 router를 모두 관측한다. 실제 block 입력과 pruner 출력을 수집하며, quantized backbone 연산량에 query dot product의 `2 × M × D` FLOPs/module을 추가한다. 원본 Cropr hook을 같이 쓴 경우 중복 계산하지 않는다.

Inference params는 dense backbone + 실제 실행 query다. D=384이면 query가 3,840개 추가된다. 학습용 aggregator와 auxiliary head는 포함하지 않는다. FP32 query payload는 15,360 bytes이며 직렬화 overhead와 분리한다.

Latency는 같은 validation image를 장치에 미리 올리고 batch 1, warmup 10, repeats 100으로 비교한다. CUDA 전후 synchronize를 유지하고 scorer/sort/gather/LLF를 포함한다. 지표 수집 hook은 timing 전에 제거한다. Fake-quant FP32 backend와 실제 정수 kernel 성능을 구분한다.

```text
eval-qat-pair/
  evaluation_config.json  # 원래 config, pruning 설정, 두 checkpoint SHA256/provenance
  split.json
  dense_metrics.json
  pruned_metrics.json    # query params, scorer precision, 실제 token trace 포함
  comparison.json        # Top-1 손실(pp), GFLOPs 비율, latency speedup
  comparison.csv
```

학습 `experiment`는 원래 값을 유지하고 평가 ID는 `evaluation_id=S-QAT-P`로 분리한다. 기존 output 덮어쓰기를 거부한다.

## 7. 완료 기준

- 원본 `CrossAttention.forward_scorer`와 score가 일치하고 Softmax/CLS self-attention을 쓰지 않는다.
- Query 미제공, shape/block mapping 불일치, 출처 불일치, 미초기화 observer는 명시적으로 거부한다.
- Disabled 경로가 기존 QAT logits와 일치하며 FP32/QAT 기존 경로는 회귀 검사로 보존한다.
- CLS 보존, 첫 17개/이후 16개 제거, 이미지별 다른 Top-K, 원본 gather 순서를 검증한다.
- 기대 trace와 최대/최종 TPR을 실제 forward에서 확인한다.
- LLF가 제거 당시 feature를 한 번씩 복원하며 반복 forward에 cache가 남지 않는다.
- 평가 중 backbone weight/observer/query가 바뀌지 않는다.
- Dense와 pruned 결과의 backbone/split hash가 같고 query hash는 별도 기록된다.
- Router parameter와 FLOPs를 포함하고 auxiliary 분기를 제외한다. 전체 forward latency에 LLF와 scorer가 포함된다.

구현 후 관련 테스트와 `python -m pytest tests A3/tests -q`로 기존 학습/재개 경로를 확인한다. 실제 query와 QAT checkpoint를 인수한 뒤 Food-101 test 비교를 수행한다. 소규모 fixture 통과를 실제 정확도나 정수 가속 검증으로 보고하지 않는다.
