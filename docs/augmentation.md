# 선택형 데이터 증강

Mixup, CutMix, RandomErasing을 **옵션으로 켜고 끌 수 있게** 한 기능이다 (이슈 [#2](https://github.com/dbwofla11/OnDiviceAI_vitQAT/issues/2)).

- **목적**: 파일럿 baseline 정확도(약 60%)가 낮으면 양자화·pruning 효과를 해석하기 어렵다. 낮은 원인은 아직 미확정이며, 이 기능은 증강을 쓸 수 있는 도구를 미리 준비한 것이다.
- **기본값은 전부 꺼짐**이라 옵션을 주지 않으면 기존 실험과 똑같이 동작한다.
- **데이터 수는 늘지 않는다.** 75,750장은 증강된 숫자가 아닌 Food-101 원본이고, 증강은 매 epoch마다 원본에 무작위 변형을 새로 적용하는 방식이다. epoch당 이미지 수와 학습 비용은 그대로다.

## 옵션

| 옵션 | 기본값 | 하는 일 |
|---|---|---|
| `--mixup A` | 0 (끔) | 두 이미지를 반투명하게 섞고 라벨도 비율대로 섞는다 |
| `--cutmix A` | 0 (끔) | 사각형 영역을 다른 이미지로 교체하고 라벨은 면적 비율로 섞는다 |
| `--random-erasing P` | 0 (끔) | 확률 P로 사각형 영역을 노이즈로 지운다 (라벨 불변) |
| `--crop-min-scale S` | 0.08 | RandomResizedCrop의 최소 면적 비율. 키울수록 crop이 약해진다 |

`--mixup`과 `--cutmix`를 함께 주면 배치마다 절반 확률로 하나가 선택된다.

## 옵션 선택 방법

- **켜기**: 학습 명령에 옵션을 추가한다. 옵션마다 독립적이라 원하는 것만 골라 쓴다.
- **끄기**: 옵션을 생략한다 (또는 값을 0으로 준다). 기본 실행이 곧 "증강 추가 없음"이다.
- **다른 실험에 적용**: `A1.train`, `A2.train`, `B2.train` 등 해당 실험 진입점에 같은 증강 옵션을 전달한다. 증강 옵션은 모델·QAT 설정과 무관하다.

| 조합 | 옵션 | 비고 |
|---|---|---|
| 기본 | (없음) | 기존 동작 |
| 표준 (DeiT 레시피) | `--mixup 0.8 --cutmix 1.0 --random-erasing 0.25` | 기존 RandAugment 등에 더해진다 |
| Mixup/CutMix만 | `--mixup 0.8 --cutmix 1.0` | 라벨을 섞는 증강만 |
| 이미지 변형만 | `--random-erasing 0.25` | 라벨을 안 섞어서 상대적으로 약하다 |
| crop만 약하게 | `--crop-min-scale 0.25` | 값은 예시이며 조정이 필요하다 |

```powershell
# 같은 증강 설정으로 FP32 / QAT / Cropr+QAT를 실행 (모드만 바꾼다)
python -m A1.train --mode fp32 --output runs/aug_fp32 --mixup 0.8 --cutmix 1.0 --random-erasing 0.25
python -m A2.train --mode qat  --output runs/aug_qat  --mixup 0.8 --cutmix 1.0 --random-erasing 0.25
python -m B2.train --mode qat --use-cropr --output runs/aug_cropr --mixup 0.8 --cutmix 1.0 --random-erasing 0.25
```

**무엇을 켤지 정하는 기준** (일반적인 경향이며 이 레포에서 검증된 값은 아니다):

- 파일럿에서 **train 정확도는 높은데 val이 낮으면**(과적합) 증강을 켠다. 보통 Mixup/CutMix의 효과가 가장 크다.
- **train도 낮거나 곡선이 아직 오르는 중이면**(미수렴) 증강보다 epoch를 늘리는 것이 먼저다.
- train/val 곡선은 output 폴더의 `epochs.jsonl`에서 확인한다. 강도는 아래 "값 가이드"를 따른다.

## 값 가이드

**표준값의 출처**: `0.8 / 1.0 / 0.25`는 DeiT 학습 레시피의 기본값이고, 이 레포의 Cropr 원본 분류 학습 코드
([cropr/cls/main.py](../cropr/cls/main.py)의 `--mixup 0.8`, `--cutmix 1.0`, `--reprob 0.25`)도 같은 값이다.
즉 "ViT 학습의 사실상 기본값이자 Cropr 원본과 같은 설정"이지, **Food-101에 맞춰 튜닝된 값은 아니다.**
(원본은 RandAugment도 timm의 `rand-m9-mstd0.5-inc1`을 쓰는데, 이 레포는 torchvision 기본 RandAugment라 완전히 같지는 않다.)

**값의 의미와 조정 방향**

| 옵션 | 표준 | 값이 의미하는 것 | 약하게 | 강하게 |
|---|---|---|---|---|
| `--mixup` | 0.8 | 섞는 비율이 Beta(α, α)를 따른다. 1보다 작으면 한쪽 이미지가 우세한 경우가 많고, 클수록 반반에 가깝다 | 0.2 ~ 0.4 | 올릴 근거 없음 |
| `--cutmix` | 1.0 | 교체되는 면적 비율이 0~100%에서 균등하게 나온다 | 0 (끔) | 올릴 근거 없음 |
| `--random-erasing` | 0.25 | 이미지 4장 중 1장꼴로 사각형을 지운다. 지우는 면적은 이미지의 2~33%로 고정 | 0 (끔) | 0.5 |
| `--crop-min-scale` | 0.08 | 클수록 원본에 가까운 영역만 자른다 (crop이 약해진다) | 0.25 ~ 0.35 | 더 낮출 근거 없음 |

"약하게"와 "강하게" 열의 값은 **제안일 뿐 검증된 값이 아니다.** Mixup α 0.2~0.4와 RandomErasing 0.5는 관련 논문에서
쓰였다고 알고 있으나 이 레포에서 확인한 것은 아니다.

**실행 순서 제안**

1. **표준값(0.8 / 1.0 / 0.25)으로 먼저 학습한다.** 원본 Cropr와 같은 설정이라 비교가 쉽고, 튜닝 부담이 없다.
2. 결과를 `epochs.jsonl`로 본다.
   - val이 baseline보다 오르고 train/val 격차가 줄면 그대로 쓴다.
   - train loss가 높은 채 정체되고 val도 낮으면(과소적합) **약하게** 바꾼다. 한 번에 하나씩 낮추는 순서는
     `--random-erasing 0` → `--mixup 0.2~0.4` → `--cutmix 0` 이다.
3. epoch 예산이 300보다 많이 짧거나 QAT가 낮은 비트에서 잘 안 되면 처음부터 약한 쪽에서 시작해도 된다.
   (강한 증강은 긴 학습이 필요하고 저비트 QAT는 학습이 어렵다는 일반론에 근거한 판단이며 검증되지 않았다.)

## 주의사항

- **비교 실험은 모두 같은 증강 설정**을 쓴다. 한쪽에만 적용하면 양자화 효과와 증강 효과가 섞인다.
- **val/test에는 증강이 적용되지 않는다.**
- Mixup/CutMix를 켜면 손실이 soft target 기준이라 **`train_loss`를 증강을 끈 실험과 비교할 수 없다.** 비교는 val/test Top-1으로 한다.
- 배치 크기가 홀수인 마지막 배치는 섞지 않고 label smoothing만 적용한다 (timm Mixup이 짝수 배치만 지원).
- Cropr 보조 loss도 같은 soft target으로 계산된다. Cropr 원본 학습 방식과 달라지므로 결과 해석 시 유의한다.
- 증강 설정이 다른 checkpoint로 resume하면 오류가 난다. 증강 옵션이 생기기 전에 저장한 옛 checkpoint는 "증강을 끄고 학습한 것"으로 취급하므로, 증강을 끈 채로만 이어 학습할 수 있다 (켜려고 하면 오류).
- 옵션 값이 잘못되면(음수, 범위 초과, NaN/inf) 학습 시작 전에 어떤 옵션이 문제인지 알려 주는 오류가 난다.
- 강한 증강이 저비트 QAT(기본 W2/A2) 수렴을 어렵게 할 수 있다는 것은 일반론에 근거한 **가설**이며 검증되지 않았다. 강도는 실험으로 정한다.

## 코드 위치

| 파일 | 역할 |
|---|---|
| [utils/augment.py](../utils/augment.py) | 증강 로직 (transform 조립, Mixup 생성, 배치 증강, 옵션 값 검사, 옵션 기본값) |
| [utils/data.py](../utils/data.py) | train transform만 `augment.py`로 교체 |
| 각 실험 `train.py` (예: [A2](../A2/train.py)) | 해당 `train_epoch`에서 배치 증강·criterion 선택 |
| [utils/checkpoint.py](../utils/checkpoint.py) | 실험이 제공한 구형 설정 기본값으로 checkpoint 비교 |
| 각 실험 `config.py` (예: [A2](../A2/config.py)) | 증강 CLI 정의·검증; 추론 CLI에는 학습 증강 인자를 노출하지 않음 |
| [utils/constants.py](../utils/constants.py) | 클래스 수(101) 공용 상수 |

## 확인 방법

```powershell
# 자동 테스트
python -m pytest tests/test_augment.py

# 그림으로 확인 (--image를 쓰면 Food-101 없이 임의의 사진으로 실행)
python scripts/check_augment.py --image photo1.jpg photo2.jpg --out augment_check `
    --mixup 0.8 --cutmix 1.0 --random-erasing 0.25
```

`same_image.png`(같은 원본을 여러 번 증강)와 `mixed_batch.png`(Mixup/CutMix 적용 배치)가 저장되고, 터미널에는 배치 각 이미지의 soft target 상위 2개 클래스가 출력된다.

## 검증 현황

| 항목 | 상태 |
|---|---|
| 자동 테스트 (CPU 전체 59개 통과·2개 skip, GPU RTX 4060에서 `test_cuda.py`와 증강 테스트 21개 통과) | 완료 |
| FP32 / QAT / Cropr+QAT 스모크 실행 | 완료 |
| 실제 사진 5장으로 시각 확인 | 완료 |
| **실제 Food-101 이미지로 시각 확인** | **미확인** |
| **증강이 정확도에 미치는 효과** | **미확인** (학습 실험 필요) |

검증은 레포 고정 버전(torch 2.3.0, timm 1.0.3)에서 했다. 다른 timm 버전에서는 Mixup 관련 함수 시그니처가 다를 수 있다.
