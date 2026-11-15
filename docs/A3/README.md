> **Current implementation (2026-10-27):** The following is the previous learned-query design and paper notes. A3 inference now selects tokens using CLS self-attention Top-K, retaining the Cropr removal schedule and LLF. No query checkpoint is required. See [execution](execution.md).

# QAT-only ViT에 추론 Cropr 적용

## A3 폴더 경계

실험 진입점과 평가 테스트는 `A3/`, 문서는 `docs/A3/`에 둔다. 모델 생성과 QAT는 `A3/model.py`, `A3/quantization.py`에서 관리하고 checkpoint I/O는 `utils`에서 가져온다. A3 전용 pruning과 지표는 `A3/pruning/`, `A3/metrics.py`에서 관리한다. 아래 learned-query 내용은 이전 설계 기록이며 현재 구현은 위 링크를 따른다.

전체 폴더 배치와 독립 관리 규칙은 [실험 폴더 구조](../experiment-folder-structure.md)를 따른다.

작성일: 2026-10-26 · 기준 코드: `abc637a`에서 확장 · 상태: **추론 평가 코드 구현, 실제 query 확보 방식 미결정**

FP32 학습과 QAT-only 학습은 다른 담당자가 수행한다. 이 작업은 QAT-only checkpoint를 받아 추론에 Cropr router를 연결하고, 같은 backbone의 pruning off/on 성능을 비교한다. Backbone 재학습은 담당 범위에 포함하지 않는다.

**설정 상속 원칙:** 이번 실험의 모델, 데이터셋, 입력/전처리, class mapping, pooling, 초기화, 학습 하이퍼파라미터, QAT 정밀도·구현, 데이터 분할과 평가 조건은 **앞선 2번 QAT-only 실험의 실제 설정을 그대로 따른다.** 이번 실험에서 추가하는 것은 Cropr 추론 pruning 설정이다. 본문에 적힌 ViT-S/16·Food-101·W4A8 및 학습 기본값은 예상 예시이며, 선행 실험의 최종 config/checkpoint가 우선한다. 논문의 원래 모델·데이터에 맞추기 위해 선행 설정을 바꾸지 않는다.

## 문서 구성

| 문서 | 관리 내용 |
|---|---|
| [논문 기준 pruning 파라미터](pruning-parameters.md) | 논문 근거, scorer/selector 구성, R 및 위치, LLF, query 인수 조건 |
| [실험 설계](experiment-design.md) | 역할 분담, 입력 산출물, 비교군, 고정 조건과 평가 방법 |
| [구현 아키텍처](architecture.md) | 모듈 책임, checkpoint/router 로드, forward와 지표, 구현 검증 |
| [실행 방법](execution.md) | 실제 CLI, query 파일 형식, 설정 상속과 산출물 |
| [라이브러리 API](../library-api.md) | import/attach, 학습·추론 모드, auxiliary loss와 query export |

**주의사항 / 합의 사항**은 [실험 설계 8절](experiment-design.md)에 정리했다. 같은 비교군의 증강 일치, Top-1 비교 기준, query 학습 경계, RNG/구형 checkpoint 검증 범위, fake-quant 측정의 해석을 확인한다.

## 확인된 결정

- Model 및 QAT 조건: 앞선 QAT-only 실험을 따른다. 현재 예상 예시는 ViT-S/16, 224 입력, 101 classes, W4A8이다.
- Scorer: **학습된 query와 token feature의 내적**. CLS attention top-k는 Cropr 자체 방식이 아니므로 기존 설계에서 제외했다.
- Schedule: 1~10번 block 뒤, R=16, 첫 모듈은 R+1=17개 제거. 총 161 patch 제거, 최대 TPR 82.14%.
- LLF: on, 11번 block 뒤에 제거 feature를 복원해 12번 block에 전체 tokens를 넣는다.
- 추론에는 router만 사용하고 aggregator/auxiliary head는 실행하지 않는다.
- 논문 본문의 R=16은 ViT-B/16 설정이다. Small/16 적용은 같은 12-block 구조를 사용하는 프로젝트 선택이다.

**필수 선행 조건:** QAT-only checkpoint에는 Cropr query가 없으므로, 논문대로 pruning하려면 별도로 학습된 query가 필요하다. Query 제공 경로 또는 별도 scorer 학습 범위를 결정해야 한다. 추론 평가 코드는 구현했으며 실제 task 학습과 성능 평가는 아직 실행하지 않았다.

이전 문서의 CLS attention scorer, 총 160개 제거, parameter 추가 없음 설계는 논문 확인 후 위 조건으로 교체했다. 학습 조건은 전달받은 config를 기록하고, 연구 조건 변경은 실험 설계에, 인터페이스 변경은 아키텍처에 반영한다.

## 참고 자료

- [Token Cropr PDF](../../papers/05_qat_token_pruning/2025_Bergner_Token_Cropr.pdf): 3.1~3.5절, 4.1절, 표 1·4.
- [중간발표](../../papers/온디바이스AI%20중간발표.pptx): 발표 17번의 CLS attention baseline과 Cropr 자체 방식은 구분한다.
- [기존 실행 문서](../experiments.md), [증강 옵션](../augmentation.md).
