# B1: Cropr + Linear QAT (LSQ/LSQ+)

모델 생성은 `B1/model.py`, optimizer·loss·학습 정책은 `train.py`,
checkpoint 추론 정책은 `inference.py`에서 독립적으로 관리한다.

```powershell
python -m B1.train --data-root data --output B1/runs/train
python -m B1.inference --eval-checkpoint B1/runs/train/best.pt --data-root data --output B1/runs/eval
```

새 학습에 `--pretrained`를 추가하면 timm의 사전학습 가중치에서 시작한다. 기본은 scratch다.
`--init-checkpoint`는 같은 구조의 FP32 checkpoint(예: A1의 `best.pt`)로 새 학습을 초기화한다.
resume/추론에서는 저장된 state를 복원하므로 사전학습 가중치나 init checkpoint를 다시 읽지 않는다.
재개할 때는 원래 학습 옵션과 output에 `--resume B1/runs/train/last.pt`를 추가한다.

원본 Cropr ViT의 transformer block Linear(qkv, proj, fc1, fc2)를 `W{weight-bits}A{activation-bits}`(기본 W2A2)로 양자화한다.
patch embedding·classifier head·Cropr scorer는 8bit로 양자화한다. attention 행렬곱은 FP32다.
weight는 출력 채널별 대칭 LSQ(기본 LSQ+ 초기화)이고, activation은 tensor별 LSQ+(비대칭, 학습 가능한 offset)다.
학습과 추론은 같은 quantizer를 사용하므로 양자화 방식이 같다. 양자화 구현은 `quantizers.py`, `modules.py`에 있다.

비교군 간 model·초기화·bit·seed·증강·학습 조건을 맞춘다.
[코드 소유권과 확장 방법](../experiment-folder-structure.md), [학습 조건·지표](../experiments.md).
