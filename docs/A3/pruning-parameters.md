> **Current implementation (2026-10-27):** The following is the previous learned-query design and paper notes. A3 inference now selects tokens using CLS self-attention Top-K, retaining the Cropr removal schedule and LLF. No query checkpoint is required. See [execution](execution.md).

# Token Cropr 논문 기준 pruning 파라미터

확인일: 2026-10-26. [논문 PDF](../../papers/05_qat_token_pruning/2025_Bergner_Token_Cropr.pdf)의 PDF 3~5, 8쪽(인쇄 페이지 9742~9744, 9747), 식 (1)~(7), 표 1·4와 저장소의 원본 구현을 확인했다.

## 논문과 동일한가: 근거 대조표

**실험 조건의 기준은 앞선 QAT-only 실험이다.** 모델·데이터셋·pooling·입력/전처리·학습 recipe·QAT·평가 조건은 선행 실험의 실제 config/checkpoint를 그대로 따른다. 아래 표에서 논문과 다른 태스크/모델 조건은 의도적으로 상속한 조건이며, 이를 논문 설정으로 바꿀 필요는 없다. ViT-S/16·Food-101·W4A8은 현재 예상 예시다. 논문에서는 Cropr의 pruning 연산과 schedule 규칙을 가져온다.

**판정: 전체 실험은 논문과 동일하지 않다.** Router 연산과 schedule 규칙을 가져와 QAT-only Small 모델에 적용하는 확장 설계다. 아래의 '일치'는 설정의 일치다. 구현은 합성 fixture로 원본 Cropr 출력과의 동등성을 검사하며, 실제 task 성능 재현과는 구분한다.

페이지는 PDF의 1부터 센 페이지다. 논문에 직접 적힌 값, 규칙을 적용해 계산한 값, 원본 코드에서 확인한 세부 사항을 구분했다.

| 항목 | 논문 내용과 근거 | 이번 설계 | 판정 |
|---|---|---|---|
| Scorer | 3쪽 3.1절 식 (3)·(4): learnable queries와 key의 내적 후 query 축 합산. 4쪽 3.4절 식 (7): 합산 query와 key의 내적으로 단순화 | 학습된 query와 각 block 출력 token의 내적 | 연산 설계 일치, query 확보 필요 |
| Query 수 | 4쪽 3.2절 Image classification: N=1 | 각 모듈 1개 | 일치 |
| Scorer 구조 | 3쪽 3.1절 및 8쪽 표 4a: single head, QKV/head projection 및 scorer LN 생략한 simple 구조 | head 1, projection/LN 없음 | 일치 |
| Selector | 3쪽 식 (1)·(2): K=M-R개 유지. 8쪽 표 4b: sampling보다 Top-K 선택 | score 상위 K 유지 | 일치 |
| CLS 보존 | 5쪽 3.5절: CLS는 제거하지 않음 | CLS score를 +inf로 고정 | 규칙 일치. +inf 구현은 원본 cropr.py 185행 |
| Pruning 위치 | 5쪽 3.5절: LLF on이면 마지막 두 block을 제외한 각 block 뒤 | 12 blocks의 1~10번 뒤 | 논문 규칙을 12 blocks에 적용 |
| R=16 적용 모델 | 5쪽 4.1절 Cropr at scale: ViT-B/16, L/16, H/14에 R=16,8,8 | ViT-S/16에 R=16 | 모델 변경. Small의 직접 보고값 아님 |
| 첫 모듈 R+1 | 5쪽 3.5절: CLS를 사용하는 경우 첫 모듈 제거량을 1 증가 | 첫 17개, 이후 16개 | 일치. 원본 vision_transformer.py 37~39행도 확인 |
| TPR | 5쪽 4.1절: ViT-B/16에서 약 82%. 같은 쪽 3.5절의 첫 단계 +1을 함께 적용 | (17+9×16)/196=82.14% | 규칙에서 산출한 정확한 값, 논문 측정 결과 아님 |
| LLF 시점 | 4쪽 3.3절: penultimate block 출력과 제거 token을 합쳐 마지막 block에서 처리 | 11번 뒤 복원, 12번에서 처리 | 일치 |
| LLF 순서 | 4쪽 3.3절은 공간 위치 복원으로 설명. 분류 원본 vision_transformer.py 103행은 keep와 각 단계 removed를 concatenate | 분류 원본 concatenate 순서 | 분류 코드와 일치, 본문 공간 순서 설명과 구분 |
| 추론 graph | 4쪽 3.4절·그림 3: aggregator와 auxiliary head 제거, router만 유지 | router만 실행 | 일치 |
| Query 학습 | 3쪽 3.1절·그림 2·식 (5), 4쪽 3.2절: aggregator와 보조 분류 loss로 query 학습, scorer 입력 stop-gradient. 8쪽 표 4c·4d에서 MLP/stop-gradient 비교 | query 학습 경로 아직 미결정 | 미충족. QAT-only checkpoint만으로 재현 불가 |
| Backbone 학습 | 5쪽 4.1절: pretrained MAE ViT-L을 ImageNet에 finetuning하며 pruning 사용 | 앞선 QAT-only 실험의 학습 결과를 그대로 사용 | 선행 실험 상속. 학습 중 pruning 적응은 추가하지 않음 |
| Pooling | 5쪽 표 1 및 caption: Cropr는 avg, CLS를 요구하는 방법은 cls | 앞선 QAT-only 모델의 pooling 유지(CLS 예상) | 선행 실험 상속 |
| Dataset | 5쪽 4.1절: ImageNet-1k | 앞선 실험의 dataset/split 사용(Food-101 예상) | 선행 실험 상속 |
| Quantization | 제공 본문의 Cropr에는 QAT/bit 설정이 없음 | 앞선 실험의 QAT 설정 유지(W4A8 예상), router FP32 제안 | Backbone은 상속, router 정밀도만 추가 조건 |
| 측정 조건 | 5쪽 4절: AMP, NVIDIA A100, batch별 최적 throughput | 앞선 실험의 평가 조건을 따르고 같은 환경에서 dense/pruned 비교 | 선행 실험 상속. 논문의 speedup 수치 직접 재현 아님 |

따라서 보고서에는 **'Token Cropr의 router 및 pruning schedule을 선행 QAT-only 모델에 적용한 추론 실험'**이라고 적는다. ViT-S/16 W4A8은 현재 예상 조건이다. 'Token Cropr 논문과 동일한 전체 실험' 또는 '논문 성능 재현 완료'라고 적지 않는다. 현재 상태는 '추론 코드 구현 및 합성 fixture 검증, 실제 query 인수와 task 평가 전'이다.

## 1. 논문에서 실제로 하는 일

각 ViT block의 출력 token을 Cropr router에 넣는다. Router는 별도로 학습된 query로 token별 task relevance 점수를 계산하고 높은 점수의 token만 다음 block에 보낸다. **ViT의 CLS self-attention을 읽는 방식과 다르다.** 논문의 Attn Top-K는 비교 기준선이며 Cropr와 별도 방법이다(4절, 표 1).

분류에서는 각 Cropr 모듈마다 query 하나를 사용한다. Projection과 scorer LayerNorm을 생략한 단순 구성에서는 현재 token feature `X`가 곧 key이고 점수는 다음과 같다.

```text
X: [B, M, D]       # 현재 block 출력, CLS 포함
q: [D]            # 해당 block용 학습된 query
score[b, j] = dot(q, X[b, j])
score[b, CLS] = +inf
keep = 점수가 높은 M - R_i개 token
removed = 나머지 R_i개 token
```

식 (3)·(4)·(7)의 raw dot product를 사용한다. 추론 router에는 Softmax, probability fake quantization, CLS attention head 평균이 필요하지 않다. 학습용 aggregator에서만 `softmax(A / sqrt(D))`로 feature를 가중 평균한다. 여러 query를 쓰는 task에서는 추론 전에 query를 합칠 수 있지만, 이번 분류 조건은 처음부터 N=1이다.

## 2. 적용 파라미터

| 항목 | 이번 설정 | 근거 / 구분 |
|---|---|---|
| Backbone | 전달받은 ViT-S/16 QAT-only, W4A8 | 프로젝트 조건, Cropr 논문은 QAT를 다루지 않음 |
| 입력 / patch / blocks | 224×224 / 16×16 / 12 | 프로젝트 조건 |
| Embedding D | 384, 전달 checkpoint 구조로 재확인 | ViT-S 구조, query 차원도 D와 일치해야 함 |
| Scorer | learned-query cross-attention dot product | 3.1절 식 (3)·(4), 3.4절 식 (7) |
| Selector | score 내림차순 Top-K, sampling 없음 | 3.1절 식 (1), 표 4b |
| Pruning 위치 | 1~10번 block **뒤** | LLF on: 마지막 두 block 뒤는 생략, 3.5절 |
| 기본 제거량 R | 16 tokens/module | 12-block ViT-B/16에 쓴 R을 Small/16에 적용, 4.1절 |
| 실제 제거량 | 첫 모듈 17, 나머지 9개 모듈 각 16 | CLS를 고려해 첫 R을 1 늘림, 3.5절 |
| CLS | 항상 유지, 항상 sequence 첫 token | 3.5절 및 원본 `scores[:, 0] = inf` |
| Query 수 | 모듈마다 1, 총 10개 | 분류 N=1, 3.2절 |
| Cropr attention heads | 1 | 3.1절, 표 4a, 원본 기본값 |
| Query / key / value projection | 모두 없음, Identity | 3.1절, 표 4a, 원본 기본값 |
| Scorer 사전 LayerNorm | 없음 | 3.1절, 표 4a |
| LLF | on, 11번 block 뒤 / 12번 block 앞 복원 | 3.3·3.5절 |
| 복원 feature | 제거된 시점의 feature를 cache에 보관 | 3.3절, 이후 중간 block 계산 생략 |
| 추론 aggregator / auxiliary head | 실행하지 않음 | 3.4절, 그림 3 |
| Pruning 수 | 이미지마다 일정한 개수 | static schedule, 2·3.5절 |
| 제거할 token의 identity | 이미지별 점수에 따라 다름 | Top-K router |
| Router 정밀도 | FP32 제안, QAT backbone 상태 유지 | 프로젝트 확장 선택. 논문이 정한 양자화 설정이 아님 |
| Backbone pooling | 전달받은 CLS pooling 유지 | 동일 checkpoint 비교 조건. 논문 표 1의 Cropr avg pooling과 다름 |

**모델 적용 범위:** 논문 본문 4.1절은 ViT-B/16, L/16, H/14에 대해 R=16, 8, 8을 제시한다. ViT-S/16의 R=16이 본문에 직접 보고된 값이라고 쓰지 않는다. 이번에는 같은 12-block 구조에 ViT-B의 schedule을 적용하는 것이다. 제공 PDF는 본문과 참고문헌 11쪽이며 부록 C의 전체 학습 하이퍼파라미터는 포함하지 않는다.

## 3. 정확한 토큰 schedule

초기 196 patch + CLS 1개 = 197 tokens. `R_i = [17,16,16,16,16,16,16,16,16,16]`이다.

| Block (1-based) | 입력 총 tokens | 뒤에서 제거하는 patch | 제거 직후 총 tokens |
|---|---:|---:|---:|
| 1 | 197 | 17 | 180 |
| 2 | 180 | 16 | 164 |
| 3 | 164 | 16 | 148 |
| 4 | 148 | 16 | 132 |
| 5 | 132 | 16 | 116 |
| 6 | 116 | 16 | 100 |
| 7 | 100 | 16 | 84 |
| 8 | 84 | 16 | 68 |
| 9 | 68 | 16 | 52 |
| 10 | 52 | 16 | 36 |
| 11 | 36 | 0 | 36, 이후 LLF로 197 복원 |
| 12 | 197 | 0 | 197 |

- 총 제거 patch: `17 + 9 × 16 = 161`.
- 최소 유지 patch: `196 - 161 = 35`, CLS 포함 36 tokens.
- 최대 patch TPR: `161 / 196 = 82.142857%`.
- 마지막 block 입력 TPR: LLF로 복원하므로 0%.
- Block 입력 trace: `[197,180,164,148,132,116,100,84,68,52,36,197]`.
- Schedule ID: `cropr-r16-first17-llf`.

3.5절은 첫 단계 +1을 설명하므로 기존 문서의 매번 정확히 16개, 총 160개 제거 설정을 폐기한다. 본문의 약 82%와 구체적인 구현 trace를 구분해 기록한다.

## 4. LLF와 원본 구현의 순서

논문 3.3절은 제거 token을 원래 공간 위치에 복원한다고 설명한다. 저장소 분류 원본은 각 단계 score 순으로 keep/removed를 나눈 후, 11번 block 뒤 `[현재 keep, 단계1 removed, ..., 단계10 removed]`로 concatenate한다. 원래 grid 순서로 scatter하지 않는다.

이번 분류 구현은 저장소 원본의 concatenate 순서를 따른다. CLS는 첫 위치에 유지하고 모든 patch가 정확히 한 번 포함되도록 확인한다. 초기 positional embedding은 이미 feature에 들어 있으므로 재추가하지 않는다. 공간 위치가 필요한 dense task로 확장하면 원본 patch ID를 사용한 spatial restoration을 별도로 구현해야 한다.

원본 selector는 `argsort(descending=True, stable=False)`를 사용한다. 따라서 동점의 원본 index 우선순위는 논문/코드가 보장하는 규칙이 아니다. 우선 원본 동작을 유지하고, 재현을 위해 stable tie-break를 도입한다면 구현 변경으로 따로 기록한다.

## 5. 학습된 query가 필요한 이유와 인수 조건

논문은 auxiliary head의 loss로 query를 학습한다. Top-K 자체로 query에 학습 신호를 주는 방식이 아니다. 학습 중에는 scorer/aggregator 입력에 stop-gradient를 적용하고, aggregator와 auxiliary classification head를 실행한다(3.1·3.2절, 표 4c·4d).

| 학습 구성 | 논문 / 원본 기준 | 이번 담당 범위 |
|---|---|---|
| Query | 각 모듈의 학습 가능한 `[1,1,D]` tensor | 학습 완료된 값을 인수해야 함 |
| Aggregator | Softmax attention 가중 평균 + LN/MLP/residual | 추론에서 제거 |
| Aggregator MLP | on, 원본 MLP ratio 4 | 학습 자료 확인용 |
| Auxiliary head | LayerNorm + Linear, 분류 CE | Food-101에 적용하면 출력 101, 추론에서 제거 |
| Stop-gradient | on | query 학습 방식 확인용 |
| Auxiliary loss 계수 | 저장소 adapter 기본 1.0 | 제공 본문에서 이 수치를 검증한 것은 아님 |
| 마지막 block DropPath | LLF 학습 시 off | 전달 backbone 재학습 없음, eval에서는 전체 DropPath off |

**일반적인 QAT-only checkpoint에는 Cropr query가 없다.** 따라서 그 checkpoint만 받아서는 학습 없이 논문 방식의 Cropr router를 완성할 수 없다. Backbone의 CLS embedding이나 무작위 query를 대신 쓰면 논문 방식의 학습된 scorer를 재현한 것으로 보고할 수 없다.

이 작업에서는 backbone 재학습을 하지 않는다. 우선 같은 QAT backbone 표현에 맞춰 학습된 10개 query와 그 학습 이력을 확보해야 한다. 다른 backbone에서 배운 query는 차원이 같아도 호환 성능이 보장되지 않으므로 출처와 결합 방법을 검증해야 한다.

현재 미결정 사항은 query 확보 방식이다.

1. 같은 고정 QAT backbone에 맞춰 학습된 router를 별도 산출물로 제공받는다.
2. 별도 범위 합의 후 backbone·observer를 고정하고 query/auxiliary branch만 학습한다. 이는 **추가 scorer 학습**이며, 논문의 전체 학습 절차와 다른 프로젝트 변형이다. Epoch/LR/feature 수집 및 pruning 노출 정책을 별도 설계해야 한다.

어느 방식을 택하든 query의 학습 이력이 있는 조건을 순수한 “추가 학습 전혀 없음”으로 쓰지 않는다. Query 학습을 허용하지 않고 제공 query도 없다면, 논문의 Attn Top-K 등 학습이 필요 없는 **별도 baseline**만 가능하다. 사용자 결정 전에는 임의로 query 학습이나 baseline 대체를 수행하지 않는다.

## 6. 인수·결과에 남길 값

Backbone checkpoint SHA256, 코드 commit, model/bit 수, split hash, class mapping 외에 다음을 추가한다.

- Query/router checkpoint SHA256, 학습에 사용한 backbone SHA256, query shape와 block mapping.
- Query 학습 데이터/split, backbone·observer 고정 여부, 학습 시 pruning 유무, loss/학습 조건.
- `scorer=learned-query-dot-product`, `num_queries=1`, `num_heads=1`, projection/LN off, router precision.
- `R=16`, `first_stage_extra=1`, block 위치 1~10, LLF on, 실제 token trace와 최대/최종 TPR.
- Pooling 유지 여부, 원본 argsort/LLF 결합 순서, score 동점 처리의 변경 여부.

Query는 ViT-S D=384일 때 `10 × 384 = 3,840`개 FP32 parameter, 순수 payload 15,360 bytes를 추가한다. Aggregator/head를 제외한 router만의 수치다. 총 parameter 수가 dense와 같다고 쓰지 않는다. Scorer dot product 비용도 FLOPs에 포함하고, sorting/gather/cache/LLF는 전체 forward latency에 포함한다.

## 7. 원본 코드 확인 위치

- [vision_transformer.py](../../cropr/cls/vision_transformer.py): schedule 첫 단계 +1, LLF on의 `depth-2`개 모듈, 마지막 block 앞 concatenate.
- [cropr.py](../../cropr/cls/cropr.py): `forward_scorer`, CLS 보존, score 내림차순 선택, 학습 aggregator 및 stop-gradient.
- [main.py](../../cropr/cls/main.py): Cropr query/head/projection/LN/MLP 기본 옵션.
- [B2/cropr_adapter.py](../../B2/cropr_adapter.py): learned-query Cropr 실험의 생성 옵션과 원본 연결.
