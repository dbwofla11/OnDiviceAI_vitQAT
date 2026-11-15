# B1: FP32 Cropr (PTQ 미구현)

모델 생성은 `B1/model.py`, optimizer·loss·학습 정책은 `train.py`,
checkpoint 추론 정책은 `inference.py`에서 독립적으로 관리한다.

```powershell
python -m B1.train --data-root data --output B1/runs/train
python -m B1.inference --eval-checkpoint B1/runs/train/best.pt --data-root data --output B1/runs/eval
```

새 학습에 `--pretrained`를 추가하면 timm의 사전학습 가중치에서 시작한다. 기본은 scratch다.
resume/추론에서는 저장된 state를 복원하므로 사전학습 가중치를 다시 받지 않는다.
재개할 때는 원래 학습 옵션과 output에 `--resume B1/runs/train/last.pt`를 추가한다.

FP32 학습·추론이다.
B1의 후속 PTQ는 아직 구현하지 않았다.

비교군 간 model·초기화·bit·seed·증강·학습 조건을 맞춘다.
[코드 소유권과 확장 방법](../experiment-folder-structure.md), [학습 조건·지표](../experiments.md).
