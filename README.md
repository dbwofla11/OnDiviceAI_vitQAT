# OnDiviceAI_vitQAT

Food-101 ViT의 양자화와 token pruning 비교 실험.

| 폴더 | 실험 |
|---|---|
| [A1](docs/A1/README.md) | FP32 dense 기준선 |
| [A2](docs/A2/README.md) | QAT-only |
| [A3](docs/A3/execution.md) | QAT 후 CLS attention pruning |
| [A4](docs/A4/README.md) | CLS attention pruning + QAT 공동 학습 |
| [B1](docs/B1/README.md) | FP32 Cropr 학습 (후속 PTQ 미구현) |
| [B2](docs/B2/README.md) | Cropr + QAT 공동 학습 |

각 실험은 `model.py`, `train.py`, `inference.py`와 자체 양자화·pruning 구현을 소유한다.
`utils/`에는 데이터·checkpoint·계측·반복 실행 같은 공통 기반 기능만 둔다.
[폴더 구조와 이전 경로 안내](docs/experiment-folder-structure.md),
[공통 학습 조건·지표](docs/experiments.md), [TensorRT export/측정](docs/tensorrt-int8.md).

```powershell
conda env create -f environment.yml
conda activate vitqat
python -m A2.train --data-root data --output A2/runs/seed42
python -m A2.inference --eval-checkpoint A2/runs/seed42/best.pt --data-root data --output A2/runs/eval
python -m pytest tests A3/tests A4/tests B2/tests -q
```

Python 의존성은 [requirements.txt](requirements.txt) 하나로 관리한다. `environment.yml`도 이 파일을 참조한다.
